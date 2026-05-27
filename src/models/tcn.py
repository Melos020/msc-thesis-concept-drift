"""Temporal Convolutional Network (thesis Ch 4.3).

3 residual blocks, dilations {1, 2, 4}, kernel 3, 64 channels, weight_norm.
Binary classification head: logit -> sigmoid -> P(up).
Default input_dim is 5 at runtime (inferred from data shape); the class
accepts any value, the default below is documentary only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel, padding=0, dilation=dilation))

    def forward(self, x):
        # Left-pad for causality
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout=0.15):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel, dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel, dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        return F.relu(x + residual)


class TCN(nn.Module):
    """3-block TCN with dilations {1, 2, 4}, 64 channels."""
    def __init__(self, input_dim: int = 4, channels: int = 64,
                 kernel: int = 3, dilations=(1, 2, 4), dropout=0.15):
        super().__init__()
        self.input_proj = weight_norm(nn.Conv1d(input_dim, channels, 1))
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, kernel, d, dropout) for d in dilations
        ])
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        # Take last timestep
        x = x[:, :, -1]
        return self.head(x).squeeze(-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = TCN(input_dim=4)
    print(f'TCN params: {count_params(m):,}')
    x = torch.randn(8, 30, 4)
    y = m(x)
    print(f'Output: {y.shape}')
