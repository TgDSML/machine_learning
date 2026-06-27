from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn


MATH_EXPR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MATH_EXPR_DIR.parent
OUTPUT_ROOT = MATH_EXPR_DIR / "extension_outputs"
FIGURES_DIR = OUTPUT_ROOT / "figures"
RESULTS_DIR = OUTPUT_ROOT / "results"
MPL_CONFIG_DIR = OUTPUT_ROOT / "matplotlib"


def configure_output_environment() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


class SnakeActivation(nn.Module):
    """Small self-contained Snake-style activation used by the experiments."""

    def __init__(self, features: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.clamp_min(1e-6)
        return x + torch.sin(alpha * x).pow(2) / alpha


class FunctionApproximator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            SnakeActivation(64),
            nn.Linear(64, 64),
            SnakeActivation(64),
            nn.Linear(64, 64),
            SnakeActivation(64),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SmallFunctionApproximator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            SnakeActivation(32),
            nn.Linear(32, 32),
            SnakeActivation(32),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DoubleSin(nn.Module):
    def __init__(self, small: bool = True) -> None:
        super().__init__()
        model_class = SmallFunctionApproximator if small else FunctionApproximator
        self.model1 = model_class()
        self.model2 = model_class()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model1(x) + self.model2(2 * x)

    def get_internal(self) -> list[nn.Module]:
        return [self.model1, self.model2]
