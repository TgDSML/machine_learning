from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from math_expr.expression_graph import parse_program_to_tree, tree_to_edges, tree_to_string
except ImportError:  # pragma: no cover - supports running from this directory
    from expression_graph import parse_program_to_tree, tree_to_edges, tree_to_string


EXAMPLES = [
    "sin(), exp()",
    "sin(), poly([0,3]), exp()\nsinc()",
    "poly( [0, 0.2] )",
]


def main() -> None:
    for program in EXAMPLES:
        tree = parse_program_to_tree(program)
        print("Program:")
        print(program)
        print()
        print("Expression:")
        print(tree_to_string(tree))
        print()
        print("Edges:")
        for parent, child in tree_to_edges(tree):
            print(f"{parent} -> {child}")
        print()


if __name__ == "__main__":
    main()
