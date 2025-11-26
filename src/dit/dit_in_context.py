# dit_in_context.py
"""
Variante DiT con In-Context Conditioning.

Según el paper:
- Se añaden dos tokens extra al input: embeddings de tiempo y clase.
- Se usan bloques Transformer estándar (no se modifican).
- Al final se descartan los tokens de condición y sólo se usan los
  tokens de imagen para la capa de salida.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_core import (
    DiTBase,
    Attention,
)


class InContextBlock(nn.Module):
    """
    Transformer block estándar:

        x = x + MSA(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
        )

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class InContextFinalLayer(nn.Module):
    """
    Capa final estándar: LN + Linear.
    """

    def __init__(self,
                 hidden_size: int,
                 patch_size: int,
                 out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear(x)
        return x


class DiT_InContext(DiTBase):
    """
    DiT con in-context conditioning:

    - x_tokens: tokens de imagen (B, N, D)
    - cond_tokens: [t_emb, y_emb] como 2 tokens extras
    - concatenamos: seq = [x_tokens, cond_tokens]
    - pasamos por bloques estándar
    - al final, separamos y usamos sólo los primeros N tokens.
    """

    def __init__(self,
                 *args,
                 mlp_ratio: float = 4.0,
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert self.y_embedder is not None, \
            "In-context también asume conditioning en clase."

        self.blocks = nn.ModuleList([
            InContextBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(self.depth)
        ])
        self.final_layer = InContextFinalLayer(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )

    # ------------------------------------------------------------------ #

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        t: (B,)
        y: (B,)
        """
        B = x.shape[0]

        x_tokens = self.x_embedder(x) + self.pos_embed  # (B, N, D)
        N = x_tokens.shape[1]

        t_emb = self.t_embedder(t)  # (B, D)
        y_emb = self.y_embedder(y)  # (B, D)
        cond_tokens = torch.stack([t_emb, y_emb], dim=1)  # (B, 2, D)

        seq = torch.cat([x_tokens, cond_tokens], dim=1)  # (B, N+2, D)

        for block in self.blocks:
            seq = block(seq)

        # separar tokens de imagen y de condición
        img_tokens = seq[:, :N, :]  # (B, N, D)

        x_out = self.final_layer(img_tokens)
        x_out = self.unpatchify(x_out)
        return x_out