from typing import Literal

import torch
import torch.nn as nn

from .dit_adaln_zero import DiT_AdaLNZero
from .dit_cross_attention import DiT_CrossAttention
from .dit_in_context import DiT_InContext


def create_dit(model_type: Literal["adaln", "cross", "incontext"],
               image_size: int,
               num_classes: int,
               patch_size: int = 2,
               device: torch.device = torch.device("cpu")) -> nn.Module:
    latent_size = image_size // 8

    hidden_size = 384
    depth = 12
    num_heads = 6
    patch_size = patch_size
    in_channels = 4
    class_dropout_prob = 0.1
    learn_sigma = False

    kwargs = dict(
        input_size=latent_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        learn_sigma=learn_sigma,
    )

    if model_type == "adaln":
        model = DiT_AdaLNZero(**kwargs)
    elif model_type == "cross":
        model = DiT_CrossAttention(**kwargs)
    elif model_type == "incontext":
        model = DiT_InContext(**kwargs)
    else:
        raise ValueError(f"model_type desconocido: {model_type}")

    model.to(device)
    return model