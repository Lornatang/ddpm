# Copyright (c) Lornatang. All rights reserved.
import torch
from torch import nn

from torch.nn import functional as F_torch

__all__ = [
    "Attention",
]


class Attention(nn.Module):
    def __init__(self, channels: int) -> None:
        """Initializes the Attention module.

        Args:
            channels (int): Dimension of the input.
        """
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.proj_q = nn.Conv2d(channels, channels, (1, 1), stride=(1, 1), padding=(0, 0))
        self.proj_k = nn.Conv2d(channels, channels, (1, 1), stride=(1, 1), padding=(0, 0))
        self.proj_v = nn.Conv2d(channels, channels, (1, 1), stride=(1, 1), padding=(0, 0))
        self.proj = nn.Conv2d(channels, channels, (1, 1), stride=(1, 1), padding=(0, 0))

        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.

        Notes:
            Implements standard attention mechanism:
                1. Compute Query/Key/Value projections
                2. Calculate attention weights via Q*K^T
                3. Apply softmax normalization
                4. Aggregate values with attention weights
                5. Final projection and residual connection[1,2]
        """
        batch_size, channels, height, width = x.shape

        # 1. Feature normalization
        h = self.group_norm(x)  # Stabilize feature magnitudes

        # 2. Compute Q/K/V projections
        q = self.proj_q(h)  # (B, C, H, W)
        k = self.proj_k(h)  # (B, C, H, W)
        v = self.proj_v(h)  # (B, C, H, W)

        # 3. Reshape for matrix operations
        # Flatten spatial dimensions (H,W) -> (N=H*W)
        q = q.permute(0, 2, 3, 1).view(batch_size, height * width, channels).contiguous()  # (B, N, C)
        k = k.view(batch_size, channels, height * width).contiguous()  # (B, C, N)
        v = v.permute(0, 2, 3, 1).view(batch_size, height * width, channels).contiguous()  # (B, N, C)

        # 4. Compute attention weights (Q*K^T scaled by 1/sqrt(d))
        attn_logits = torch.bmm(q, k) * (channels ** -0.5)  # (B, N, N)

        # 5. Apply softmax normalization
        attn_weights = F_torch.softmax(attn_logits, dim=-1)  # (B, N, N)

        # 6. Attention-weighted value aggregation
        h = torch.bmm(attn_weights, v)  # (B, N, C)

        # 7. Reshape back to spatial dimensions
        h = h.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # 8. Final projection and residual connection
        h = self.proj(h)

        # 9.Residual skip-connection
        out = x + h

        return out
