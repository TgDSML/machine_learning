# Expression Graph Extension

This extension adds a symbolic expression graph layer on top of the existing
`math_expr` graph language. It parses `.graph` programs into expression trees,
converts those trees into differentiable PyTorch modules, and includes examples
and experiments that demonstrate automatic differentiation through the parsed
structure.

## Architecture

- `expression_graph.py` parses the existing DSL into `Node` trees and provides
  string, edge-list, Mermaid, Graphviz, and optional NetworkX views.
- `nn_nodes.py` contains PyTorch modules for the supported operations:
  `sin`, `exp`, `ln`, `sinc`, `poly`, sum, and composition.
- `module_from_tree.py` converts parsed expression trees into `torch.nn.Module`
  instances.
- `executor_adapter.py` bridges `.graph` files to the original executor without
  modifying `executor.py`.
- `examples/` contains small executable demonstrations.
- `experiments/` contains the validation scripts used to reproduce the extension
  experiments.

Generated files are written to `math_expr/extension_outputs/`, which is ignored
by git.

## Dependencies

The extension examples and tests require:

```text
numpy
matplotlib
pillow
torch
pytest
```

Optional visualization helpers:

```text
graphviz
networkx
```

The experiments include a small self-contained Snake-style activation so they do
not require the external `snake` package.

## Run Tests

From the repository root:

```powershell
python -m pytest math_expr/test_expression_graph.py -q
```

## Run Examples

From the repository root:

```powershell
python math_expr/examples/exp_expression_graph_demo.py
python math_expr/examples/exp_expression_graph_ad.py
python math_expr/examples/exp_executor_adapter_demo.py
python math_expr/examples/exp_expression_graph_visualize.py
```

The visualization example writes Mermaid files and, when Graphviz is available,
PNG files under:

```text
math_expr/extension_outputs/figures/
```

## Run Experiments

From the repository root:

```powershell
python math_expr/experiments/baseline_sin_plus_ln.py
python math_expr/experiments/snake_sin_plus_ln_ln_focused.py
python math_expr/experiments/snake_exp_plus_exp2_exp_focused.py
```

The experiments create their output directories automatically and write figures
and metric files under:

```text
math_expr/extension_outputs/
```

These outputs are reproducible artifacts and should not be committed.
