# dit_core.py
"""
Core components for Diffusion Transformer (DiT).

Incluye:
- Attention multi-cabeza (self-attention estándar en espacio (B, N, D))
- Embeddings de tiempo y clase
- PatchEmbed estilo ViT (Conv2d con stride=patch_size)
- Positional embeddings sinusoidales 2D
- Clase base DiTBase (sin fijar el tipo de bloque)
"""

import math
from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------------

def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN modulation: x * (1 + scale) + shift
    x: (B, N, D)
    shift, scale: (B, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """
    Multi-Head Self-Attention para tensores (B, N, D).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim debe ser divisible entre num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (3, B, H, N, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeding de timesteps escalares en vectores de tamaño hidden_size.
    Usa embedding sinusoidal de dim=frequency_embedding_size seguido
    de un MLP de 2 capas con SiLU.
    """

    def __init__(self,
                 hidden_size: int,
                 frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor,
                           dim: int,
                           max_period: int = 10_000) -> torch.Tensor:
        """
        t: (B,) enteros o floats.
        return: (B, dim) embedding sinusoidal.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """
    Embeding de labels de clase y soporta dropout de tokens para
    classifier-free guidance (CFG).
    """

    def __init__(self,
                 num_classes: int,
                 hidden_size: int,
                 dropout_prob: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        # +1 para el token "null" usado en CFG
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def token_drop(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Con prob. dropout_prob reemplaza la clase por el índice num_classes.
        """
        if self.dropout_prob <= 0.0 or not self.training:
            return labels
        drop = torch.rand_like(labels.float()) < self.dropout_prob
        drop = drop.to(labels.device)
        return torch.where(drop.bool(), torch.full_like(labels, self.num_classes), labels)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        labels = self.token_drop(labels)
        return self.embedding_table(labels)


class PatchEmbed(nn.Module):
    """
    Conv2d para convertir imagen (B, C, H, W) en tokens (B, N, D),
    con kernel y stride = patch_size.
    """

    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Se espera imagen {self.img_size}x{self.img_size}, recibida {H}x{W}"
        x = self.proj(x)  # (B, D, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


# -------------------------------------------------------------------------
# Positional embeddings 2D
# -------------------------------------------------------------------------

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int,
                                       pos: np.ndarray) -> np.ndarray:
    """
    embed_dim debe ser par. pos es (M,) o (M,).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


def get_2d_sincos_pos_embed(embed_dim: int,
                            grid_size: int) -> np.ndarray:
    """
    Retorna embedding posicional 2D de tamaño (grid_size^2, embed_dim)
    usando sin/cos separables en x e y, concatenados.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, Gh, Gw)
    grid = np.stack(grid, axis=0)  # (2, Gh, Gw)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                               grid[0].reshape(-1))
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                               grid[1].reshape(-1))
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb  # (Gh*Gw, D)


# -------------------------------------------------------------------------
# Clase base del modelo (sin fijar el bloque concreto porque queremos implementar las 3 posibles variantes)
# -------------------------------------------------------------------------

class DiTBase(nn.Module):
    """
    Backbone genérico de DiT que opera sobre latentes 2D.

    Las subclases deben:
      - Definir self.blocks (lista de bloques)
      - Definir self.final_layer
      - Implementar forward (cómo entra la condición en los bloques)
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        num_classes: Optional[int] = None,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = True,
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)

        if num_classes is not None:
            self.y_embedder = LabelEmbedder(
                num_classes=num_classes,
                hidden_size=hidden_size,
                dropout_prob=class_dropout_prob,
            )
        else:
            self.y_embedder = None

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size),
            requires_grad=False,
        )

        self.blocks = nn.ModuleList()  # definido en subclases
        self.final_layer = None        # definido en subclases

        self.initialize_weights()

    # ---------------------------------------------------------

    def initialize_weights(self):
        # init lineal estándar
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.apply(_basic_init)

        # pos embeddings fijas sin-cos
        pos = get_2d_sincos_pos_embed(
            self.hidden_size,
            int(self.x_embedder.num_patches ** 0.5),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos).float().unsqueeze(0)
        )

    # ---------------------------------------------------------

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, patch_size^2 * C_out)
        return: (B, C_out, H, W)
        """
        B, N, D = x.shape
        p = self.patch_size
        C = self.out_channels
        h = w = int(math.sqrt(N))
        assert h * w == N, "N debe ser cuadrado perfecto"

        x = x.reshape(B, h, w, p, p, C)
        # (B, h, w, p, p, C) -> (B, C, h, p, w, p)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, h * p, w * p)
        return x