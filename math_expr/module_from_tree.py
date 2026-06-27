from __future__ import annotations

import torch
from torch import nn

try:
    from math_expr.expression_graph import Node
    from math_expr.nn_nodes import (
        ComposeNode,
        ExpNode,
        LnNode,
        PolyNode,
        SincNode,
        SinNode,
        SumNode,
    )
except ImportError:  # pragma: no cover - supports running examples from inside math_expr/
    from expression_graph import Node
    from nn_nodes import (
        ComposeNode,
        ExpNode,
        LnNode,
        PolyNode,
        SincNode,
        SinNode,
        SumNode,
    )


class IdentityNode(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def node_to_module(node: Node) -> nn.Module:
    """Convert an expression tree node into an equivalent PyTorch module."""
    op_type = node.op_type.upper()

    if op_type == "X":
        return IdentityNode()
    if op_type == "SUM":
        return SumNode([node_to_module(child) for child in node.children])
    if op_type == "SIN":
        return _wrap_unary(SinNode(), node)
    if op_type == "EXP":
        return _wrap_unary(ExpNode(), node)
    if op_type in {"LN", "LOG"}:
        return _wrap_unary(LnNode(), node)
    if op_type == "SINC":
        return _wrap_unary(SincNode(), node)
    if op_type == "POLY":
        return _wrap_unary(PolyNode(node.params or []), node)

    raise ValueError(f"Unsupported op_type for module conversion: {node.op_type}")


def _wrap_unary(outer: nn.Module, node: Node) -> nn.Module:
    if not node.children:
        return ComposeNode(outer, IdentityNode())
    if len(node.children) != 1:
        raise ValueError(f"{node.op_type} expects exactly one child")
    return ComposeNode(outer, node_to_module(node.children[0]))
