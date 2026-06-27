from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from _common import (
    FIGURES_DIR,
    RESULTS_DIR,
    SmallFunctionApproximator,
    configure_output_environment,
)

configure_output_environment()

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FIGURE_PATH = FIGURES_DIR / "baseline_sin_plus_ln.png"
METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
COMPARISON_PATH = RESULTS_DIR / "baseline_comparison.md"
GRAPH_GUIDED_MSE_F = 0.00035406


@dataclass(frozen=True)
class TrainingConfig:
    x_min: float = 0.1
    x_max: float = 10.0
    train_points: int = 8192
    eval_points: int = 1000
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    seed: int = 42


def build_dataset_sin_plus_ln(config: TrainingConfig) -> TensorDataset:
    x = torch.linspace(config.x_min, config.x_max, config.train_points).unsqueeze(1)
    y_f = torch.sin(x) + torch.log(x)
    return TensorDataset(x, y_f)


def train_baseline(
    model: SmallFunctionApproximator,
    train_loader: DataLoader,
    config: TrainingConfig,
) -> list[float]:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[float] = []

    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        batches = 0

        for x, y_f in train_loader:
            optimizer.zero_grad()
            f_hat = model(x)
            loss = loss_fn(f_hat, y_f)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        epoch_loss = total_loss / batches
        history.append(epoch_loss)
        print(f"Epoch {epoch:03d}/{config.epochs} loss={epoch_loss:.6f}")

    return history


def evaluate_baseline(
    model: SmallFunctionApproximator,
    config: TrainingConfig,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    x_eval = torch.linspace(config.x_min, config.x_max, config.eval_points).unsqueeze(1)

    with torch.no_grad():
        f_true = torch.sin(x_eval) + torch.log(x_eval)
        f_hat = model(x_eval)
        metrics = {"mse_f": nn.functional.mse_loss(f_hat, f_true).item()}

    return metrics, x_eval, f_true, f_hat


def plot_baseline_results(
    x_eval: torch.Tensor,
    f_true: torch.Tensor,
    f_hat: torch.Tensor,
    output_path: Path = FIGURE_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_np = x_eval.squeeze(1).cpu().numpy()
    f_true_np = f_true.squeeze(1).cpu().numpy()
    f_hat_np = f_hat.squeeze(1).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6), dpi=160)
    ax.plot(x_np, f_true_np, linewidth=2.0, label="true f(x) = sin(x) + ln(x)")
    ax.plot(x_np, f_hat_np, "--", linewidth=2.0, label="predicted f_hat(x)")
    ax.set_title("Baseline Neural Function Approximation")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metrics: dict[str, float], output_path: Path = METRICS_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def build_comparison(metrics: dict[str, float]) -> tuple[str, str]:
    baseline_mse = metrics["mse_f"]
    table = "\n".join(
        [
            "| Method                                | MSE        |",
            "| ------------------------------------- | ---------- |",
            f"| Baseline Neural Approximation         | {baseline_mse:.8f} |",
            f"| Expression Graph Guided Approximation | {GRAPH_GUIDED_MSE_F:.8f} |",
        ]
    )

    if baseline_mse > GRAPH_GUIDED_MSE_F:
        absolute_gain = baseline_mse - GRAPH_GUIDED_MSE_F
        relative_gain = absolute_gain / baseline_mse * 100.0
        interpretation = (
            "The graph-guided method improved performance by "
            f"{absolute_gain:.8f} MSE, a {relative_gain:.2f}% reduction versus the baseline."
        )
    elif baseline_mse < GRAPH_GUIDED_MSE_F:
        absolute_delta = GRAPH_GUIDED_MSE_F - baseline_mse
        relative_delta = absolute_delta / GRAPH_GUIDED_MSE_F * 100.0
        interpretation = (
            "The graph-guided method did not improve performance; the baseline was better by "
            f"{absolute_delta:.8f} MSE, a {relative_delta:.2f}% reduction "
            "versus the graph-guided result."
        )
    else:
        interpretation = "Both methods achieve similar accuracy."

    return table, interpretation


def save_comparison(
    table: str,
    interpretation: str,
    output_path: Path = COMPARISON_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{table}\n\n{interpretation}\n", encoding="utf-8")


def main() -> None:
    config = TrainingConfig()
    torch.manual_seed(config.seed)

    dataset = build_dataset_sin_plus_ln(config)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = SmallFunctionApproximator()

    print("Model:")
    print(model)
    print()
    print("Training baseline with loss = MSE(f_hat, sin(x) + ln(x))")

    train_baseline(model, train_loader, config)
    metrics, x_eval, f_true, f_hat = evaluate_baseline(model, config)
    plot_baseline_results(x_eval, f_true, f_hat)
    save_metrics(metrics)

    table, interpretation = build_comparison(metrics)
    save_comparison(table, interpretation)

    print()
    print("Evaluation metrics:")
    print(f"mse_f={metrics['mse_f']:.8f}")
    print(f"Saved figure: {FIGURE_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved comparison: {COMPARISON_PATH}")
    print()
    print(table)
    print()
    print("Interpretation:")
    print(interpretation)


if __name__ == "__main__":
    main()
