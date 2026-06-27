from pathlib import Path

import torch

import executor_adapter
from expression_graph import parse_program_to_tree, tree_to_mermaid, tree_to_string
from module_from_tree import node_to_module


def evaluate_tree_torch(node, x):
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


def test_parse_composition_to_string():
    tree = parse_program_to_tree("sin(), exp()")

    assert tree.op_type == "SIN"
    assert tree_to_string(tree) == "sin(exp(x))"


def test_multiline_expression_parses_as_sum():
    tree = parse_program_to_tree("sin(), poly([0,3]), exp()\nsinc()")

    assert tree.op_type == "SUM"
    assert len(tree.children) == 2
    assert "+" in tree_to_string(tree)


def test_poly_parameter_parsing_preserves_coefficients():
    tree = parse_program_to_tree("poly([0, 0.2, 3])")

    assert tree.op_type == "POLY"
    assert tree.params == [0, 0.2, 3]
    assert tree_to_string(tree) == "poly([0, 0.2, 3], x)"


def test_tree_to_mermaid_exports_basic_composition():
    tree = parse_program_to_tree("sin(), exp()")
    mermaid = tree_to_mermaid(tree)

    assert "graph TD" in mermaid
    assert '["SIN"]' in mermaid
    assert '["EXP"]' in mermaid
    assert '["X"]' in mermaid


def test_node_to_module_matches_recursive_torch_evaluator():
    tree = parse_program_to_tree("sin(), poly([1, 2]), exp()")
    model = node_to_module(tree)
    x = torch.linspace(-1.0, 1.0, steps=5)

    assert torch.allclose(model(x), evaluate_tree_torch(tree, x))


def test_node_to_module_supports_backward_for_sum():
    tree = parse_program_to_tree("sin(), exp()\nsinc()")
    model = node_to_module(tree)
    x = torch.linspace(-1.0, 1.0, steps=5, requires_grad=True)
    loss = model(x).mean()

    loss.backward()

    assert x.grad is not None


def test_evaluate_graph_with_tree_calls_original_executor_in_compatible_workspace(
    tmp_path,
    monkeypatch,
):
    graph_file = tmp_path / "demo.graph"
    graph_file.write_text("sin(), exp()", encoding="utf-8")
    observed = {}

    def fake_original_execute(program_name, x=None):
        observed["program_name"] = program_name
        observed["graph_exists"] = Path("graphs/demo.graph").exists()
        observed["x"] = x
        return "original-result"

    monkeypatch.setattr(executor_adapter, "original_execute", fake_original_execute)

    result, tree = executor_adapter.evaluate_graph_with_tree(graph_file, x="sample-x")

    assert observed == {
        "program_name": "demo",
        "graph_exists": True,
        "x": "sample-x",
    }
    assert result.original_return == "original-result"
    assert result.execution_successful is True
    assert tree_to_string(tree) == "sin(exp(x))"


def test_execute_with_tree_returns_executor_result_and_tree(tmp_path, monkeypatch):
    graph_file = tmp_path / "demo.graph"
    graph_file.write_text("sinc()", encoding="utf-8")

    monkeypatch.setattr(
        executor_adapter,
        "original_execute",
        lambda program_name, x=None: "executor-result",
    )

    executor_result, tree = executor_adapter.execute_with_tree(graph_file)

    assert executor_result == "executor-result"
    assert tree_to_string(tree) == "sinc(x)"
