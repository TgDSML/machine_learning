from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from _common import DoubleSin, FIGURES_DIR, RESULTS_DIR, configure_output_environment

configure_output_environment()

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from math_expr.expression_graph import Node, parse_program_to_tree, tree_to_string


FIGURE_PATH = FIGURES_DIR / "snake_exp_plus_exp2_improved.png"
METRICS_PATH = RESULTS_DIR / "snake_exp_plus_exp2_metrics.json"


@dataclass(frozen=True)
class TrainingConfig:
    x_min: float = -2.0
    x_max: float = 2.0
    train_points: int = 8192
    eval_points: int = 1000
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    lambda_exp: float = 2.0
    lambda_scaled: float = 0.5
    seed: int = 42


def build_dataset_exp_plus_exp2(config: TrainingConfig) -> TensorDataset:
    x = torch.linspace(config.x_min, config.x_max, config.train_points).unsqueeze(1)
    y_g = torch.exp(x)
    y_scaled = torch.exp(2.0 * x)
    y_f = y_g + y_scaled
    return TensorDataset(x, y_f, y_g, y_scaled)


def build_models() -> DoubleSin:
    return DoubleSin(small=True)


def parse_and_validate_expression_graph(program: str) -> Node:
    tree = parse_program_to_tree(program)
    assert tree.op_type.upper() == "SUM", f"Expected SUM root, found {tree.op_type}"
    assert len(tree.children) == 2, f"Expected two SUM branches, found {len(tree.children)}"

    exp_branch, scaled_branch = tree.children
    assert exp_branch.op_type.upper() == "EXP", (
        f"Expected EXP first branch, found {exp_branch.op_type}"
    )
    assert exp_branch.children[0].op_type.upper() == "X", (
        "Expected first branch to represent exp(x)"
    )

    assert scaled_branch.op_type.upper() == "EXP", (
        f"Expected EXP second branch, found {scaled_branch.op_type}"
    )
    scaled_inner = scaled_branch.children[0]
    assert scaled_inner.op_type.upper() == "POLY", (
        f"Expected POLY inside scaled branch, found {scaled_inner.op_type}"
    )
    assert scaled_inner.params == [0, 2], (
        f"Expected POLY [0, 2] for 2x scaling, found {scaled_inner.params}"
    )

    print("Expression graph program:")
    print(program)
    print("Parsed expression tree:", tree_to_string(tree))
    print("Top-level node:", tree.op_type.upper())
    print("Branches: exp(x), exp(poly([0, 2], x))")
    print()
    return tree


def forward_components(
    model: DoubleSin,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    branch_exp, branch_scaled = model.get_internal()
    g_hat = branch_exp(x)
    g_scaled_hat = branch_scaled(2.0 * x)
    f_hat = g_hat + g_scaled_hat
    return f_hat, g_hat, g_scaled_hat


def train_exp_plus_exp2_with_exp_aux_loss(
    model: DoubleSin,
    train_loader: DataLoader,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        totals = {
            "loss_f": 0.0,
            "loss_exp": 0.0,
            "loss_scaled": 0.0,
            "total_loss": 0.0,
        }
        batches = 0

        for x, y_f, y_g, y_scaled in train_loader:
            optimizer.zero_grad()
            f_hat, g_hat, g_scaled_hat = forward_components(model, x)

            loss_f = loss_fn(f_hat, y_f)
            loss_exp = loss_fn(g_hat, y_g)
            loss_scaled = loss_fn(g_scaled_hat, y_scaled)
            total_loss = (
                loss_f
                + config.lambda_exp * loss_exp
                + config.lambda_scaled * loss_scaled
            )

            total_loss.backward()
            optimizer.step()

            totals["loss_f"] += loss_f.item()
            totals["loss_exp"] += loss_exp.item()
            totals["loss_scaled"] += loss_scaled.item()
            totals["total_loss"] += total_loss.item()
            batches += 1

        epoch_metrics = {name: value / batches for name, value in totals.items()}
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"loss_f={epoch_metrics['loss_f']:.6f} "
            f"loss_exp={epoch_metrics['loss_exp']:.6f} "
            f"loss_scaled={epoch_metrics['loss_scaled']:.6f} "
            f"total={epoch_metrics['total_loss']:.6f}"
        )

    return history


def plot_exp_plus_exp2_results(
    model: DoubleSin,
    config: TrainingConfig,
    output_path: Path = FIGURE_PATH,
) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    x_eval = torch.linspace(config.x_min, config.x_max, config.eval_points).unsqueeze(1)
    with torch.no_grad():
        f_hat, g_hat, g_scaled_hat = forward_components(model, x_eval)
        exp_true = torch.exp(x_eval)
        exp2_true = torch.exp(2.0 * x_eval)
        f_true = exp_true + exp2_true

        metrics = {
            "mse_f": nn.functional.mse_loss(f_hat, f_true).item(),
            "mse_exp": nn.functional.mse_loss(g_hat, exp_true).item(),
            "mse_scaled": nn.functional.mse_loss(g_scaled_hat, exp2_true).item(),
        }

    x_np = x_eval.squeeze(1).cpu().numpy()
    f_true_np = f_true.squeeze(1).cpu().numpy()
    f_hat_np = f_hat.squeeze(1).cpu().numpy()
    exp_true_np = exp_true.squeeze(1).cpu().numpy()
    exp_hat_np = g_hat.squeeze(1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)

    axes[0].plot(x_np, f_true_np, linewidth=2.0, label="true f(x) = exp(x) + exp(2x)")
    axes[0].plot(x_np, f_hat_np, "--", linewidth=2.0, label="predicted f_hat(x)")
    axes[0].set_title("Composite Function Approximation")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x_np, exp_true_np, linewidth=2.0, label="true g(x) = exp(x)")
    axes[1].plot(x_np, exp_hat_np, "--", linewidth=2.0, label="predicted g_hat(x)")
    axes[1].set_title("Focused Secondary Function Approximation")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("g(x)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return metrics


def save_metrics(metrics: dict[str, float], output_path: Path = METRICS_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = TrainingConfig()
    torch.manual_seed(config.seed)

    parse_and_validate_expression_graph("exp()\nexp(), poly([0, 2])")

    dataset = build_dataset_exp_plus_exp2(config)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = build_models()

    print("Model:")
    print(model)
    print()
    print(
        "Training with total_loss = MSE(f_hat, f) "
        f"+ {config.lambda_exp} * MSE(g_hat, exp) "
        f"+ {config.lambda_scaled} * MSE(g_scaled_hat, exp(2x))"
    )

    train_exp_plus_exp2_with_exp_aux_loss(model, train_loader, config)
    metrics = plot_exp_plus_exp2_results(model, config)
    save_metrics(metrics)

    print()
    print("Evaluation metrics:")
    print(f"mse_f={metrics['mse_f']:.8f}")
    print(f"mse_exp={metrics['mse_exp']:.8f}")
    print(f"mse_scaled={metrics['mse_scaled']:.8f}")
    print(f"Saved figure: {FIGURE_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
