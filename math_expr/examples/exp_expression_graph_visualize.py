from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from math_expr.expression_graph import (
        export_tree_to_dot,
        parse_program_to_tree,
        tree_to_mermaid,
        tree_to_string,
    )
except ImportError:  # pragma: no cover - supports running from this directory
    from expression_graph import (
        export_tree_to_dot,
        parse_program_to_tree,
        tree_to_mermaid,
        tree_to_string,
    )


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "extension_outputs" / "figures"

EXAMPLES = [
    ("sin_exp", "sin(), exp()", "sin_exp_tree.png"),
    (
        "sin_poly_exp_plus_sinc",
        "sin(), poly([0,3]), exp()\nsinc()",
        "sin_poly_exp_plus_sinc_tree.png",
    ),
    ("poly", "poly([0, 0.2])", "poly_tree.png"),
]


def ad_flow_mermaid_for_sin_exp() -> str:
    return "\n".join(
        [
            "graph TD",
            '  X["x"]',
            '  EXP["exp"]',
            '  SIN["sin"]',
            '  Y["output"]',
            '  LOSS["loss"]',
            "  X --> EXP --> SIN --> Y --> LOSS",
            "  LOSS -. backward / gradients .-> Y",
            "  Y -.-> SIN",
            "  SIN -.-> EXP",
            "  EXP -.-> X",
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, program, png_name in EXAMPLES:
        tree = parse_program_to_tree(program)
        mermaid = tree_to_mermaid(tree)
        mermaid_path = OUTPUT_DIR / f"{name}_tree.mmd"

        print("Program:")
        print(program)
        print()
        print("Expression:")
        print(tree_to_string(tree))
        print()
        print("Mermaid:")
        print(mermaid)
        print()

        mermaid_path.write_text(mermaid + "\n", encoding="utf-8")

        try:
            generated_path = export_tree_to_dot(
                tree,
                str(OUTPUT_DIR / png_name),
                title=tree_to_string(tree),
            )
            print(f"Graphviz PNG: {generated_path}")
        except RuntimeError as exc:
            print(f"Warning: {exc}")
        print()

    ad_flow = ad_flow_mermaid_for_sin_exp()
    ad_flow_path = OUTPUT_DIR / "sin_exp_ad_flow.mmd"
    ad_flow_path.write_text(ad_flow + "\n", encoding="utf-8")
    print("AD flow Mermaid:")
    print(ad_flow)
    print()
    print(f"AD flow saved to: {ad_flow_path}")


if __name__ == "__main__":
    main()
