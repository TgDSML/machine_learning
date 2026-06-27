from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Node:
    """A node in a parsed mathematical expression tree."""

    op_type: str
    params: Any = None
    children: list["Node"] = field(default_factory=list)


def parse_program_to_tree(program: str) -> Node:
    """Parse a graph DSL program into an expression tree."""
    lines = [line.strip() for line in program.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Program is empty")

    line_nodes = [_parse_program_line(line) for line in lines]
    if len(line_nodes) == 1:
        return line_nodes[0]
    return Node("SUM", children=line_nodes)


def tree_to_string(node: Node) -> str:
    """Render an expression tree as a readable mathematical expression."""
    op_type = node.op_type.upper()
    if op_type == "X":
        return "x"
    if op_type == "SUM":
        return " + ".join(tree_to_string(child) for child in node.children)
    if op_type == "POLY":
        child = _single_child(node)
        return f"poly({_format_params(node.params)}, {tree_to_string(child)})"

    child = _single_child(node)
    return f"{op_type.lower()}({tree_to_string(child)})"


def tree_to_edges(node: Node) -> list[tuple[str, str]]:
    """Return expression tree edges as parent and child labels."""
    edges = []
    for child in node.children:
        edges.append((_node_label(node), _node_label(child)))
        edges.extend(tree_to_edges(child))
    return edges


def tree_to_mermaid(node: Node) -> str:
    """Render an expression tree as a Mermaid graph definition."""
    lines = ["graph TD"]
    nodes: list[tuple[str, Node]] = []
    edges: list[tuple[str, str]] = []

    def visit(current: Node, parent_id: str | None = None) -> None:
        node_id = f"N{len(nodes)}"
        nodes.append((node_id, current))
        if parent_id is not None:
            edges.append((parent_id, node_id))
        for child in current.children:
            visit(child, node_id)

    visit(node)

    for node_id, current in nodes:
        lines.append(f'  {node_id}["{_escape_mermaid_label(_node_label(current))}"]')
    for parent_id, child_id in edges:
        lines.append(f"  {parent_id} --> {child_id}")
    return "\n".join(lines)


def export_tree_to_dot(
    node: Node,
    output_path: str,
    *,
    title: str | None = None,
) -> str:
    """Export an expression tree to a Graphviz-rendered PNG file."""
    try:
        from graphviz import Digraph
        from graphviz.backend.execute import ExecutableNotFound
    except ImportError as exc:
        raise RuntimeError(
            "Graphviz export requires the optional Python package. Install it with "
            "`pip install graphviz`. The Graphviz system binary may also be needed."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    filename = (
        str(output.with_suffix(""))
        if output.suffix.lower() == ".png"
        else str(output)
    )

    graph = Digraph("expression_tree", format="png")
    graph.attr(
        rankdir="TB",
        bgcolor="white",
        pad="0.25",
        nodesep="0.45",
        ranksep="0.65",
    )
    if title:
        graph.attr(label=title, labelloc="t", fontsize="18", fontname="Helvetica")
    graph.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F7FAFC",
        color="#334155",
        fontname="Helvetica",
        fontsize="12",
        margin="0.12,0.08",
    )
    graph.attr("edge", color="#64748B", arrowsize="0.75")

    counter = 0

    def add_node(current: Node, parent_id: str | None = None) -> None:
        nonlocal counter
        node_id = f"N{counter}"
        counter += 1
        graph.node(node_id, _node_label(current))
        if parent_id is not None:
            graph.edge(parent_id, node_id)
        for child in current.children:
            add_node(child, node_id)

    add_node(node)
    try:
        return graph.render(filename=filename, cleanup=True)
    except ExecutableNotFound as exc:
        raise RuntimeError(
            "Graphviz export requires the Graphviz system binary in addition to the "
            "Python package. Install the Python package with `pip install graphviz` "
            "and install Graphviz for your operating system."
        ) from exc


def tree_to_networkx(node: Node):
    """Return a NetworkX directed graph for an expression tree, if available."""
    try:
        import networkx as nx
    except ImportError:
        return None

    graph = nx.DiGraph()

    counter = 0

    def add_node(current: Node, parent_id: str | None = None):
        nonlocal counter
        node_id = f"n{counter}"
        counter += 1
        graph.add_node(
            node_id,
            label=_node_label(current),
            op_type=current.op_type,
            params=current.params,
        )
        if parent_id is not None:
            graph.add_edge(parent_id, node_id)
        for child in current.children:
            add_node(child, node_id)

    add_node(node)
    return graph


def _parse_program_line(line: str) -> Node:
    calls = [_parse_call(part) for part in _split_top_level_commas(line)]
    if not calls:
        raise ValueError(f"No function calls found in line: {line!r}")

    current = Node("X")
    for op_type, params in reversed(calls):
        current = Node(op_type, params=params, children=[current])
    return current


def _split_top_level_commas(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            part = text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1

    part = text[start:].strip()
    if part:
        parts.append(part)
    return parts


def _parse_call(text: str) -> tuple[str, Any]:
    open_index = text.find("(")
    close_index = text.rfind(")")
    if open_index == -1 or close_index == -1 or close_index < open_index:
        raise ValueError(f"Invalid function call: {text!r}")

    fn_name = text[:open_index].strip().lower()
    raw_params = text[open_index + 1:close_index].strip()
    if fn_name == "log":
        fn_name = "ln"
    if fn_name not in {"exp", "ln", "sin", "sinc", "poly"}:
        raise ValueError(f"Unsupported function: {fn_name!r}")

    params = None
    if raw_params:
        try:
            params = ast.literal_eval(raw_params)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid parameters for {fn_name}: {raw_params!r}") from exc
    return fn_name.upper(), params


def _single_child(node: Node) -> Node:
    if len(node.children) != 1:
        raise ValueError(f"{node.op_type} expects exactly one child")
    return node.children[0]


def _node_label(node: Node) -> str:
    if node.op_type.upper() == "POLY" and node.params is not None:
        return f"POLY {_format_params(node.params)}"
    return node.op_type.upper()


def _format_params(params: Any) -> str:
    if isinstance(params, list):
        return "[" + ", ".join(str(value) for value in params) + "]"
    return str(params)


def _escape_mermaid_label(label: str) -> str:
    return label.replace("\\", "\\\\").replace('"', '\\"')
