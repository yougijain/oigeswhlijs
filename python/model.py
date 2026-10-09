"""The one model this project runs: three conv layers and a linear layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard CIFAR-10 per-channel statistics (train set, pixel/255), float32.
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)


class TinyCNN(nn.Module):
    """conv3x3(3->32) relu pool, conv3x3(32->64) relu pool, conv3x3(64->128) relu pool, fc(2048->10)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 32x32x32 -> 16x16x32
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 16x16x64 -> 8x8x64
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # 8x8x128 -> 4x4x128
        return self.fc(torch.flatten(x, 1))


def normalize(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW -> float32 normalized, exactly the arithmetic the C++ loader repeats."""
    mean = torch.tensor(NORM_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x_u8.float() / 255.0 - mean) / std


def describe(model: nn.Module) -> list[tuple[str, tuple[int, ...], int]]:
    """(name, shape, param_count) per parameter tensor."""
    return [(n, tuple(p.shape), p.numel()) for n, p in model.named_parameters()]


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def forward_flops(model: TinyCNN) -> int:
    """Multiply-adds counted as 2 FLOPs, for one 32x32x3 image. Ignores bias/relu/pool."""
    flops = 0
    h = 32
    for conv in (model.conv1, model.conv2, model.conv3):
        k = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
        flops += 2 * conv.out_channels * k * h * h
        h //= 2
    flops += 2 * model.fc.in_features * model.fc.out_features
    return flops
