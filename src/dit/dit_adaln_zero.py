# dit_adaln_zero.py
"""
DiT con bloques AdaLN-Zero (la variante principal del paper).

Usa DiTBase de dit_core y define:
- AdaLNZeroBlock
- AdaLNZeroFinalLayer
- DiT_AdaLNZero (forward con conditioning vía adaLN-Zero)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_core import (
    DiTBase,
    Attention,
    modulate,
)

class AdaLNZeroBlock(nn.Module):
    """
    Bloque DiT con adaptive LayerNorm Zero (adaLN-Zero).
    Recibe:
      x: (B, N, D)
      c: (B, D)  (embedding de tiempo + clase)
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
        )
        self.norm2 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # MLP que produce [shift_msa, scale_msa, gate_msa,
        #                  shift_mlp, scale_mlp, gate_mlp]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self,
                x: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        assert c.shape == (B, D)

        shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class AdaLNZeroFinalLayer(nn.Module):
    """
    Capa final de DiT con adaLN-Zero (norm + lin modulados).
    """

    def __init__(self,
                 hidden_size: int,
                 patch_size: int,
                 out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self,
                x: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_AdaLNZero(DiTBase):
    """
    DiT completo usando AdaLN-Zero (lo más parecido al repo oficial).
    """

    def __init__(self,
                 *args,
                 mlp_ratio: float = 4.0,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # construimos blocks y final_layer aquí
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(self.depth)
        ])
        self.final_layer = AdaLNZeroFinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )

        # Inicialización especial: última capa de adaLN a cero
        self._init_adaln_zero()

    # ------------------------------------------------------------------ #

    def _init_adaln_zero(self):
        # zero-init de la última capa de adaLN en cada bloque y capa final
        for block in self.blocks:
            last = block.adaLN_modulation[-1]
            nn.init.constant_(last.weight, 0.0)
            nn.init.constant_(last.bias, 0.0)

        last = self.final_layer.adaLN_modulation[-1]
        nn.init.constant_(last.weight, 0.0)
        nn.init.constant_(last.bias, 0.0)

        nn.init.constant_(self.final_layer.linear.weight, 0.0)
        nn.init.constant_(self.final_layer.linear.bias, 0.0)

    # ------------------------------------------------------------------ #

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                y: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, C, H, W) latentes (e.g., salida de VAE)
        t: (B,) timesteps
        y: (B,) labels de clase (opcional)
        """
        B = x.shape[0]

        # patchify + pos embedding
        x = self.x_embedder(x) + self.pos_embed  # (B, N, D)

        t_emb = self.t_embedder(t)  # (B, D)

        if self.y_embedder is not None and y is not None:
            y_emb = self.y_embedder(y)  # (B, D)
        else:
            y_emb = torch.zeros_like(t_emb)

        c = t_emb + y_emb  # condición global

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)  # (B, N, p^2 * C_out)
        x = self.unpatchify(x)      # (B, C_out, H, W)
        return x