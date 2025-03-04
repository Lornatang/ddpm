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
        self.flash = flash
        self.scale = scale

        self.attn_dropout = nn.Dropout(dropout, inplace=True)

    def flash_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Performs scaled dot-product attention using the CUDA Flash Attention implementation
        for improved performance and memory efficiency.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, heads, seq_len_q, dim_head]
            k (torch.Tensor): Key tensor of shape [batch_size, heads, seq_len_k, dim_head]
            v (torch.Tensor): Value tensor of shape [batch_size, heads, seq_len_k, dim_head]

        Returns:
            torch.Tensor: Output attention tensor of shape [batch_size, heads, seq_len_q, dim_head]
        """
        if self.scale is not None:
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        # Ensure tensors are contiguous and half in memory for optimal performance
        q, k, v = map(lambda t: t.contiguous().half(), (q, k, v))

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            out = F_torch.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.).float()

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
