from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterator


DEFAULT_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "math_expr_matplotlib"
DEFAULT_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_CONFIG_DIR))

try:
    from math_expr.executor import execute as original_execute
except ImportError:  # pragma: no cover - supports running from inside math_expr/
    from executor import execute as original_execute

try:
    from math_expr.expression_graph import Node, parse_program_to_tree
except ImportError:  # pragma: no cover - supports running from inside math_expr/
    from expression_graph import Node, parse_program_to_tree


@dataclass(frozen=True)
class ExecutorAdapterResult:
    """Execution metadata returned by the adapter wrapper."""

    graph_path: Path
    program_name: str
    original_return: Any
    execution_successful: bool


def load_program_from_graph_file(path: str | PathLike[str]) -> str:
    return Path(path).read_text(encoding="utf-8")


def evaluate_graph_with_tree(
    graph_path: str | PathLike[str],
    x=None,
) -> tuple[ExecutorAdapterResult, Node]:
    """
    Parse a .graph file to an expression tree and evaluate it with the original
    executor without modifying the existing executor implementation.

    The original executor expects to open graphs/<program>.graph relative to the
    process cwd and writes plot files as a side effect. This wrapper creates a
    temporary compatible workspace, calls the unchanged executor there, and then
    returns execution metadata together with the parsed tree.
    """
    graph_file = Path(graph_path).resolve()
    program = load_program_from_graph_file(graph_file)
    tree = parse_program_to_tree(program)

    with _original_executor_workspace(graph_file) as program_name:
        original_return = original_execute(program_name, x)

    result = ExecutorAdapterResult(
        graph_path=graph_file,
        program_name=program_name,
        original_return=original_return,
        execution_successful=True,
    )
    return result, tree


def execute_with_tree(
    graph_path: str | PathLike[str],
    x=None,
    *,
    return_tree: bool = True,
):
    """
    Read a .graph file, parse it to an expression tree, then call the original
    executor without changing its implementation.

    By default this returns (executor_result, tree). Pass return_tree=False for
    compatibility with executor-style callers that only expect the executor
    result.
    """
    result, tree = evaluate_graph_with_tree(graph_path, x)

    if return_tree:
        return result.original_return, tree
    return result.original_return


@contextmanager
def _original_executor_workspace(graph_file: Path) -> Iterator[str]:
    """Create the graphs/<program>.graph layout expected by original_execute."""
    with tempfile.TemporaryDirectory(prefix="executor_adapter_") as workspace:
        workspace_path = Path(workspace)
        graphs_dir = workspace_path / "graphs"
        mpl_config_dir = workspace_path / "matplotlib"
        graphs_dir.mkdir(parents=True)
        mpl_config_dir.mkdir(parents=True)
        shutil.copy2(graph_file, graphs_dir / graph_file.name)

        previous_cwd = Path.cwd()
        previous_mpl_config_dir = os.environ.get("MPLCONFIGDIR")
        try:
            os.chdir(workspace_path)
            os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
            yield graph_file.stem
        finally:
            os.chdir(previous_cwd)
            if previous_mpl_config_dir is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_mpl_config_dir
