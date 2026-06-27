from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from math_expr.executor_adapter import evaluate_graph_with_tree, load_program_from_graph_file
    from math_expr.expression_graph import tree_to_edges, tree_to_string
except ImportError:  # pragma: no cover - supports running from this directory
    from executor_adapter import evaluate_graph_with_tree, load_program_from_graph_file
    from expression_graph import tree_to_edges, tree_to_string


MATH_EXPR_DIR = Path(__file__).resolve().parents[1]
DEMO_GRAPH = MATH_EXPR_DIR / "graphs" / "sine_in_exp.graph"


def main() -> None:
    program = load_program_from_graph_file(DEMO_GRAPH)
    x = np.linspace(-1.0, 1.0, 5)
    result, tree = evaluate_graph_with_tree(DEMO_GRAPH, x=x)

    print("Input .graph program:")
    print(program.strip())
    print()
    print("Pretty expression:")
    print(tree_to_string(tree))
    print()
    print("Tree edges:")
    for parent, child in tree_to_edges(tree):
        print(f"{parent} -> {child}")
    print()
    print("Original executor result successfully obtained:")
    print(result.execution_successful)


if __name__ == "__main__":
    main()
