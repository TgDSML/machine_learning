from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class SinNode(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class ExpNode(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class LnNode(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)


class SincNode(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sinc(x)


class PolyNode(nn.Module):
    def __init__(self, coeffs: Sequence[float] | None):
        super().__init__()
        self.coeffs = [float(coefficient) for coefficient in (coeffs or [])]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for power, coefficient in enumerate(self.coeffs):
            result = result + coefficient * x.pow(power)
        return result


class SumNode(nn.Module):
    def __init__(self, children: list[nn.Module]):
        super().__init__()
        self.terms = nn.ModuleList(children)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(child(x) for child in self.terms)


class ComposeNode(nn.Module):
    def __init__(self, outer: nn.Module, inner: nn.Module):
        super().__init__()
        self.outer = outer
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outer(self.inner(x))
