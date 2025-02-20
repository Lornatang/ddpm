# Copyright (c) Lornatang. All rights reserved.
import torch
from torch import nn
from torch.nn import functional as F_torch
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = [
    "Attend",
]


class Attend(nn.Module):
    def __init__(
            self,
            dropout: float = 0.,
            flash: bool = False,
            scale: float = None
    ) -> None:
        """Initializes the Attend module.

        Args:
            dropout (float, optional): Dropout rate. Defaults to 0.
            flash (bool, optional): Whether to use flash attention. Defaults to False.
            scale (float, optional): Scaling factor for attention. Defaults to None.
        """
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash

    def flash_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes flash attention.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output tensor after applying flash attention.
        """
        if self.scale is not None:
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F_torch.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)

        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Attend module.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        if self.flash:
            return self.flash_attn(q, k, v)

        scale = q.shape[-1] ** -0.5 if self.scale is None else self.scale

        # similarity
        sim = torch.einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
