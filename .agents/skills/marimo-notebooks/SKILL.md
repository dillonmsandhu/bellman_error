---
name: marimo-notebooks
description: Author, refactor, and validate reactive Marimo Python notebooks (.py). Enforces single-assignment rules to prevent variable collision errors (CELL NOT RUN), guides UI layout patterns (mo.ui, mo.hstack, mo.vstack), and provides automated validation scripts.
---

# Marimo Notebook Authoring and Validation

This skill provides best practices, architectural rules, and an automated validation protocol for writing and maintaining [Marimo](https://marimo.io) reactive Python notebooks (`.py`).

---

## 1. Core Architecture & The Single-Assignment Rule

Marimo notebooks are **pure Python files** executed as a **Directed Acyclic Graph (DAG)** of cells, rather than sequential stateful steps.

### The Single-Assignment Rule
* Every global variable in a Marimo notebook must be defined by **exactly one cell**.
* If two different cells define the same variable name (for example, `for algo_key in algos:` in two different cells), Marimo blocks execution with:
  ```text
  CELL NOT RUN
  This cell redefines variables from other cells.
  'algo_key' was also defined by: cell-4
  Fix: Wrap in a function
  ```

### Private Variables (The `_` Prefix)
* Any variable name beginning with an underscore (e.g. `_df`, `_idx`, `_algo_key`, `_ax`) is treated by Marimo as **cell-private**.
* Private variables are **not** published to the notebook's global DAG.
* **Rule**: Always prefix loop iterators, indices, temporary dataframes, and scratch variables with `_`.

```python
# ❌ BAD: 'item' and 'i' become global symbols and will collide with other cells!
@app.cell
def _(data_list):
    results = []
    for i, item in enumerate(data_list):
        results.append(item * 2)
    return results

# ✅ GOOD: Loop variables are private, only 'results' is published to the DAG.
@app.cell
def _(data_list):
    results = []
    for _i, _item in enumerate(data_list):
        results.append(_item * 2)
    return results
```

### Encapsulation Pattern
For complex data transformations, plotting routines, or animations, encapsulate intermediate computation in helper functions:

```python
@app.cell
def _(ALGOS, runs_data, plt, np):
    def _create_comparison_plot():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for _idx, (_k, _v) in enumerate(ALGOS.items()):
            ...
        return fig

    fig_comparison = _create_comparison_plot()
    return (fig_comparison,)
```

---

## 2. Notebook Structure & File Template

Every Marimo notebook should follow this standard template:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy",
#     "matplotlib",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import marimo as mo

    return mo, np, os, pd, plt, sys


@app.cell
def _(mo):
    mo.md("# Title\nDescription of the analysis.")
    return


@app.cell
def _(mo):
    # UI Controls
    slider = mo.ui.slider(start=1, stop=100, value=10, label="Parameter")
    return (slider,)


@app.cell
def _(mo, slider):
    # Reactive consumer cell
    mo.md(f"Current value: **{slider.value}**")
    return


if __name__ == "__main__":
    app.run()
```

---

## 3. Interactive UI & Layout Patterns

### Common Widgets
* `mo.ui.text(value="...", label="...")`
* `mo.ui.dropdown(options=[...], value="...", label="...")`
* `mo.ui.slider(start=..., stop=..., step=..., value=..., label="...")`
* `mo.ui.number(start=..., stop=..., value=..., label="...")`
* `mo.ui.checkbox(value=False, label="...")`
* `mo.ui.button(label="...")`
* `mo.ui.table(df)`

### Composing Layouts
* `mo.hstack([widget_a, widget_b], justify="start", gap=2)`: Places elements horizontally side-by-side.
* `mo.vstack([heading, content], gap=1)`: Stacks elements vertically.
* `mo.as_html(fig)` or `plt.gca()`: Converts matplotlib figures cleanly into Marimo HTML output without calling `plt.show()`.
* `mo.image(src=filepath_or_bytes)`: Embeds image or GIF files directly in the layout.

---

## 4. Automated Validation Protocol

Always run the automated validator before completing any task that creates or modifies a Marimo notebook:

```bash
python .agents/skills/marimo-notebooks/scripts/validate.py <path_to_notebook>.py
```

### What the Validator Checks
1. **Python Syntax**: Verifies syntax using `py_compile`.
2. **App Import**: Ensures the file defines a valid Marimo `app` object.
3. **Symbol Collisions**: Scans all cells via Marimo's cell manager AST and flags any shared, non-private variable names that cause `CELL NOT RUN` errors.

### Quick Manual Smoke Test
You can also inspect defined cell exports directly from the command line:
```bash
python -c "
from <notebook_module> import app
for i, cell in enumerate(app._cell_manager.cells()):
    print(f'Cell {i}: defs={sorted(list(cell.defs))}')
"
```
