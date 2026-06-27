from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from math_expr.expression_graph import Node, parse_program_to_tree, tree_to_string
    from math_expr.module_from_tree import node_to_module
except ImportError:  # pragma: no cover - supports running from this directory
    from expression_graph import Node, parse_program_to_tree, tree_to_string
    from module_from_tree import node_to_module


def evaluate_tree_torch(node: Node, x: torch.Tensor) -> torch.Tensor:
    op_type = node.op_type.upper()
    if op_type == "X":
        return x
    if op_type == "SUM":
        return sum(evaluate_tree_torch(child, x) for child in node.children)

    child_value = evaluate_tree_torch(node.children[0], x)
    if op_type == "SIN":
        return torch.sin(child_value)
    if op_type == "EXP":
        return torch.exp(child_value)
    if op_type in {"LN", "LOG"}:
        return torch.log(child_value)
    if op_type == "SINC":
        return torch.sinc(child_value)
    if op_type == "POLY":
        result = torch.zeros_like(child_value)
        for power, coefficient in enumerate(node.params or []):
            result = result + float(coefficient) * child_value.pow(power)
        return result

    raise ValueError(f"Unsupported operation for torch evaluation: {node.op_type}")


def main() -> None:
    program = "sin(), exp()"
    tree = parse_program_to_tree(program)
    model = node_to_module(tree)

    x = torch.linspace(-1.0, 1.0, steps=5, requires_grad=True)
    y = model(x)
    loss = (y**2).mean()
    loss.backward()

    print("Expression:", tree_to_string(tree))
    print("Loss:", loss.item())
    print("x.grad:", x.grad)
    print("Gradients flow through the module graph that mirrors the expression tree.")


if __name__ == "__main__":
    main()
