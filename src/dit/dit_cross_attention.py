# dit_cross_attention.py
"""
Variante DiT con bloques de Cross-Attention.

Según el paper:
- Se forma una secuencia de longitud 2 con [t_emb, y_emb].
- Cada bloque tiene self-attention sobre los tokens de imagen,
  seguido de una capa de cross-attention donde las queries son
  los tokens de imagen y keys/values vienen de los dos tokens
  de condición.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_core import (
    DiTBase,
    Attention,
)


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention:
      queries = x  (B, N, D)
      keys,values = context (B, M, D)  (M=2 en nuestro caso)
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        context: (B, M, D)
        """
        B, N, C = x.shape
        _, M, _ = context.shape

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # (B, H, N, Dh)

        kv = (
            self.kv(context)
            .reshape(B, M, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (2, B, H, M, Dh)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Bloque estándar de Transformer + cross-attention adicional.

        x = x + MSA(LN(x))
        x = x + CrossAttn(LN(x), cond_tokens)
        x = x + MLP(LN(x))
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
        )

        self.norm_ca = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(
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

    def forward(self,
                x: torch.Tensor,
                cond_tokens: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)  imagen
        cond_tokens: (B, 2, D)  [t_emb, y_emb]
        """
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm_ca(x), cond_tokens)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionFinalLayer(nn.Module):
    """
    Capa final simple: LN + Linear.
    La condición ya fue inyectada vía cross-attention.
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


class DiT_CrossAttention(DiTBase):
    """
    DiT completo con bloques de cross-attention.
    """

    def __init__(self,
                 *args,
                 mlp_ratio: float = 4.0,
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert self.y_embedder is not None, \
            "Cross-attention tiene sentido para modelo condicional en clase."

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(self.depth)
        ])
        self.final_layer = CrossAttentionFinalLayer(
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
        x = self.x_embedder(x) + self.pos_embed  # (B, N, D)

        t_emb = self.t_embedder(t)  # (B, D)
        y_emb = self.y_embedder(y)  # (B, D)
        cond_tokens = torch.stack([t_emb, y_emb], dim=1)  # (B, 2, D)

        for block in self.blocks:
            x = block(x, cond_tokens)

        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x