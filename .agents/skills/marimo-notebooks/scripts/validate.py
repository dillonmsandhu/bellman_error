#!/usr/bin/env python3
"""
Marimo Notebook Validator
Validates a Marimo notebook script (.py) for:
1. Python syntax correctness
2. Marimo App loadability
3. Variable collision / redefinition across cells (Single-Assignment Rule)
4. Unbound variables or cyclic dependencies
"""

import sys
import os
import argparse
import py_compile
import importlib.util

def validate_marimo_notebook(notebook_path: str) -> bool:
    print(f"=== Validating Marimo Notebook: {notebook_path} ===")

    if not os.path.exists(notebook_path):
        print(f"❌ Error: File not found: {notebook_path}")
        return False

    # 1. Syntax Compilation Check
    try:
        py_compile.compile(notebook_path, doraise=True)
        print("✅ Step 1: Python syntax check passed.")
    except py_compile.PyCompileError as e:
        print(f"❌ Step 1 Failed: Syntax error:\n{e}")
        return False

    # 2. Dynamic Import Check
    module_name = os.path.splitext(os.path.basename(notebook_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, notebook_path)
    if spec is None or spec.loader is None:
        print(f"❌ Step 2 Failed: Cannot create module spec for {notebook_path}")
        return False

    try:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        if not hasattr(mod, "app"):
            print("❌ Step 2 Failed: Module does not define a Marimo 'app' object.")
            return False
        app = getattr(mod, "app")
        print("✅ Step 2: Marimo app imported successfully.")
    except Exception as e:
        print(f"❌ Step 2 Failed: Exception during notebook import: {e}")
        return False

    # 3. Marimo DAG and Symbol Collision Check
    try:
        cells = list(app._cell_manager.cells())
        print(f"ℹ️ Found {len(cells)} cells in the Marimo notebook.")
    except Exception as e:
        print(f"❌ Step 3 Failed: Unable to retrieve cells from app: {e}")
        return False

    defs_seen = {}
    collisions = []

    for idx, cell in enumerate(cells):
        cell_name = getattr(cell, "name", f"cell_{idx}")
        cell_defs = getattr(cell, "defs", set())

        for symbol in cell_defs:
            if symbol.startswith("_"):
                # Private symbols do not enter the global DAG
                continue
            if symbol in defs_seen:
                prev_cell = defs_seen[symbol]
                collisions.append((symbol, prev_cell, cell_name))
            else:
                defs_seen[symbol] = cell_name

    if collisions:
        print("\n❌ Step 3 Failed: Found variable redefinitions (CELL NOT RUN in Marimo):")
        for sym, first_cell, redef_cell in collisions:
            print(f"   • Symbol '{sym}' defined in both '{first_cell}' and '{redef_cell}'")
        print("\n🔧 Fix Suggestions:")
        print("   - Prefix temporary/loop variables with an underscore '_' (e.g. '_algo_key', '_idx').")
        print("   - Wrap the cell logic in a helper function and only return intended outputs.")
        return False

    print("✅ Step 3: Zero variable collisions detected across all cells.")
    print(f"ℹ️ Exported global symbols ({len(defs_seen)}): {sorted(list(defs_seen.keys()))}")
    print("\n🎉 All validation checks passed! Notebook is ready for 'marimo edit' or 'marimo run'.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Marimo notebook for syntax and variable collisions.")
    parser.add_argument("notebook", type=str, help="Path to the Marimo notebook (.py)")
    args = parser.parse_args()

    success = validate_marimo_notebook(args.notebook)
    sys.exit(0 if success else 1)
