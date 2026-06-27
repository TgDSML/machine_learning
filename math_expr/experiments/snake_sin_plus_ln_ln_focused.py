from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from _common import FIGURES_DIR, SmallFunctionApproximator, configure_output_environment

configure_output_environment()

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from math_expr.expression_graph import Node, parse_program_to_tree, tree_to_string


FIGURE_PATH = FIGURES_DIR / "snake_sin_plus_ln_improved.png"


@dataclass(frozen=True)
class TrainingConfig:
    x_min: float = 0.1
    x_max: float = 10.0
    train_points: int = 8192
    eval_points: int = 1000
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    lambda_ln: float = 2.0
    seed: int = 42


class SinPlusLnModel(nn.Module):
    """Two small approximators summed as f_hat = g1_hat + g2_hat."""

    def __init__(self) -> None:
        super().__init__()
        self.g1 = SmallFunctionApproximator()
        self.g2 = SmallFunctionApproximator()

    def forward_components(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g1_hat = self.g1(x)
        g2_hat = self.g2(x)
        f_hat = g1_hat + g2_hat
        return f_hat, g1_hat, g2_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_hat, _, _ = self.forward_components(x)
        return f_hat


def build_dataset_sin_plus_ln(config: TrainingConfig) -> TensorDataset:
    x = torch.linspace(config.x_min, config.x_max, config.train_points).unsqueeze(1)
    y_sin = torch.sin(x)
    y_ln = torch.log(x)
    y_f = y_sin + y_ln
    return TensorDataset(x, y_f, y_sin, y_ln)


def build_models() -> SinPlusLnModel:
    return SinPlusLnModel()


def parse_and_validate_expression_graph(program: str) -> Node:
    tree = parse_program_to_tree(program)
    child_ops = [child.op_type.upper() for child in tree.children]

    assert tree.op_type.upper() == "SUM", f"Expected SUM root, found {tree.op_type}"
    assert child_ops == ["SIN", "LN"], f"Expected SIN and LN children, found {child_ops}"

    print("Expression graph program:")
    print(program)
    print("Parsed expression tree:", tree_to_string(tree))
    print("Top-level node:", tree.op_type.upper())
    print("Children:", ", ".join(child_ops))
    print()
    return tree


def train_sin_plus_ln_with_ln_aux_loss(
    model: SinPlusLnModel,
    train_loader: DataLoader,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        totals = {"loss_f": 0.0, "loss_ln": 0.0, "loss_g1": 0.0, "total_loss": 0.0}
        batches = 0

        for x, y_f, y_sin, y_ln in train_loader:
            optimizer.zero_grad()
            f_hat, g1_hat, g2_hat = model.forward_components(x)

            loss_f = loss_fn(f_hat, y_f)
            loss_ln = loss_fn(g2_hat, y_ln)
            loss_g1 = loss_fn(g1_hat, y_sin)
            total_loss = loss_f + config.lambda_ln * loss_ln

            total_loss.backward()
            optimizer.step()

            totals["loss_f"] += loss_f.item()
            totals["loss_ln"] += loss_ln.item()
            totals["loss_g1"] += loss_g1.item()
            totals["total_loss"] += total_loss.item()
            batches += 1

        epoch_metrics = {name: value / batches for name, value in totals.items()}
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"loss_f={epoch_metrics['loss_f']:.6f} "
            f"loss_ln={epoch_metrics['loss_ln']:.6f} "
            f"loss_g1={epoch_metrics['loss_g1']:.6f} "
            f"total={epoch_metrics['total_loss']:.6f}"
        )

    return history


def plot_sin_plus_ln_results(
    model: SinPlusLnModel,
    config: TrainingConfig,
    output_path: Path = FIGURE_PATH,
) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    x_eval = torch.linspace(config.x_min, config.x_max, config.eval_points).unsqueeze(1)
    with torch.no_grad():
        f_hat, g1_hat, g2_hat = model.forward_components(x_eval)
        f_true = torch.sin(x_eval) + torch.log(x_eval)
        ln_true = torch.log(x_eval)
        sin_true = torch.sin(x_eval)

        metrics = {
            "mse_f": nn.functional.mse_loss(f_hat, f_true).item(),
            "mse_ln": nn.functional.mse_loss(g2_hat, ln_true).item(),
            "mse_g1": nn.functional.mse_loss(g1_hat, sin_true).item(),
        }

    x_np = x_eval.squeeze(1).cpu().numpy()
    f_true_np = f_true.squeeze(1).cpu().numpy()
    f_hat_np = f_hat.squeeze(1).cpu().numpy()
    ln_true_np = ln_true.squeeze(1).cpu().numpy()
    ln_hat_np = g2_hat.squeeze(1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)

    axes[0].plot(x_np, f_true_np, linewidth=2.0, label="true f(x) = sin(x) + ln(x)")
    axes[0].plot(x_np, f_hat_np, "--", linewidth=2.0, label="predicted f_hat(x)")
    axes[0].set_title("Composite Function Approximation")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x_np, ln_true_np, linewidth=2.0, label="true g2(x) = ln(x)")
    axes[1].plot(x_np, ln_hat_np, "--", linewidth=2.0, label="predicted g2_hat(x)")
    axes[1].set_title("Focused Secondary Function Approximation")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("g2(x)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Evaluation metrics:")
    print(f"mse_f={metrics['mse_f']:.8f}")
    print(f"mse_ln={metrics['mse_ln']:.8f}")
    print(f"mse_g1={metrics['mse_g1']:.8f}")
    print(f"Saved figure: {output_path}")
    return metrics


def main() -> None:
    config = TrainingConfig()
    torch.manual_seed(config.seed)

    parse_and_validate_expression_graph("sin()\nln()")

    dataset = build_dataset_sin_plus_ln(config)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = build_models()

    print("Model:")
    print(model)
    print()
    print(
        "Training with total_loss = MSE(f_hat, f) "
        f"+ {config.lambda_ln} * MSE(g2_hat, ln)"
    )

    train_sin_plus_ln_with_ln_aux_loss(model, train_loader, config)
    plot_sin_plus_ln_results(model, config)


if __name__ == "__main__":
    main()
