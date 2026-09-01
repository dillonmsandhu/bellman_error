# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo>=0.17.6",
#     "matplotlib>=3.7.0",
#     "numpy>=1.24.0",
#     "pandas>=2.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="normal")


@app.cell
def _():
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.path.abspath('.')
    notebooks_dir = current_dir if os.path.basename(current_dir) == 'notebooks' else os.path.join(current_dir, 'notebooks')
    repo_root = os.path.abspath(os.path.join(notebooks_dir, '..'))
    for p in [repo_root, notebooks_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Force JAX to CPU to prevent GPU VRAM exhaustion when unpickling sweep files
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"

    import json
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo
    try:
        from notebooks.analyze_sweeps import (
            load_sweep_data,
            extract_best_configuration,
            find_latest_run_dir,
            discover_algorithm_sweeps,
        )
    except ImportError:
        from analyze_sweeps import (
            load_sweep_data,
            extract_best_configuration,
            find_latest_run_dir,
            discover_algorithm_sweeps,
        )

    return (
        discover_algorithm_sweeps,
        extract_best_configuration,
        find_latest_run_dir,
        load_sweep_data,
        mo,
        np,
        os,
        pd,
        plt,
    )


@app.cell
def _():
    TASKS = [
        {"id": "fixed_mountaincar", "name": "Fixed Policy - MountainCar-v0", "policy": "fixed", "env": "MountainCar-v0"},
        {"id": "fixed_fourrooms", "name": "Fixed Policy - FourRooms-misc", "policy": "fixed", "env": "FourRooms-misc"},
        {"id": "random_mountaincar", "name": "Random Policy - MountainCar-v0", "policy": "random", "env": "MountainCar-v0"},
        {"id": "random_fourrooms", "name": "Random Policy - FourRooms-misc", "policy": "random", "env": "FourRooms-misc"},
    ]

    EXACT_ALGOS = ["exact_td", "exact_mc", "exact_E_gd", "exact_td_lambda", "exact_td_symmetric", "exact_Etd"]
    SAMPLED_ALGOS = ["td", "sampled_E", "monte_carlo", "mc", "td0"]

    ALGO_DISPLAY_NAMES = {
        "exact_td": "Exact TD",
        "exact_mc": "Exact MC",
        "exact_E_gd": "Exact E (GD)",
        "exact_E": "Exact E (GD)",
        "exact_Etd": "Exact E + TD",
        "exact_E_td": "Exact E + TD",
        "exact_td_lambda": "Exact TD(λ)",
        "exact_td_symmetric": "Exact TD (Sym)",
        "td": "Sampled TD(0)",
        "sampled_E": "Sampled E",
        "monte_carlo": "Sampled MC",
        "mc": "Sampled MC",
        "td0": "Sampled TD(0) Clipped",
    }

    ALGO_COLORS = {
        "exact_td": "#1f77b4",        # Blue
        "exact_mc": "#ff7f0e",        # Orange
        "exact_E_gd": "#2ca02c",      # Green
        "exact_E": "#2ca02c",         # Green
        "exact_Etd": "#8c564b",       # Brown
        "exact_E_td": "#8c564b",      # Brown
        "exact_td_lambda": "#d62728", # Red
        "exact_td_symmetric": "#9467bd", # Purple
        "td": "#1f77b4",              # Blue
        "sampled_E": "#2ca02c",        # Green
        "monte_carlo": "#ff7f0e",     # Orange
        "mc": "#ff7f0e",              # Orange
        "td0": "#9467bd",             # Purple
    }
    return ALGO_COLORS, ALGO_DISPLAY_NAMES, EXACT_ALGOS, SAMPLED_ALGOS, TASKS


@app.cell
def _(
    TASKS,
    discover_algorithm_sweeps,
    find_latest_run_dir,
    load_sweep_data,
    os,
):
    def load_all_task_data(
        base_results_dir="results",
        custom_batches=None,
        custom_paths=None,
    ):
        """
        Discovers and loads sweep data across the 4 core tasks.

        Args:
            base_results_dir: Root results directory (default "results").
            custom_batches: List or string of batch directory paths, e.g.:
                ["results/fixed/sweeps/fixed_MountainCar-v0_20260828_094649", ...]
            custom_paths: Dict mapping (policy, env) or task_id to batch dirs or algo paths.
        """
        all_task_data = {t["id"]: {"task_info": t, "runs": {}} for t in TASKS}

        # 1. If custom_batches provided (list or multi-line string)
        if custom_batches:
            if isinstance(custom_batches, str):
                custom_batches = [line.strip() for line in custom_batches.strip().splitlines() if line.strip() and not line.strip().startswith("#")]
            for batch_dir in custom_batches:
                if not os.path.exists(batch_dir):
                    print(f"Warning: Custom batch directory not found: {batch_dir}")
                    continue
                # Inspect algorithms inside batch directory
                for item in os.listdir(batch_dir):
                    item_path = os.path.join(batch_dir, item)
                    if not os.path.isdir(item_path):
                        continue

                    # Could be algo/tuning or direct algo dir
                    tuning_dir = os.path.join(item_path, "tuning")
                    search_dir = tuning_dir if os.path.exists(tuning_dir) else item_path
                    ts, env, run_path = find_latest_run_dir(search_dir)
                    if not run_path:
                        continue

                    try:
                        sweep = load_sweep_data(run_path)
                        env_name = sweep.get("env_name", env)
                        policy_type = "fixed" if "fixed" in batch_dir else ("ppo" if "ppo" in batch_dir else "random")
                        # Match to task
                        for task in TASKS:
                            if task["policy"] == policy_type and (task["env"] == env_name or env_name in task["env"] or task["env"] in batch_dir):
                                all_task_data[task["id"]]["runs"][item] = sweep
                    except Exception as e:
                        print(f"Error loading {item} from {batch_dir}: {e}")

        # 2. If custom_paths provided (explicit mapping)
        if custom_paths:
            for key, paths in custom_paths.items():
                target_tasks = []
                for t in TASKS:
                    if key == t["id"] or key == (t["policy"], t["env"]):
                        target_tasks.append(t["id"])
                if isinstance(paths, str):
                    paths = [paths]
                for p in paths:
                    for algo in os.listdir(p) if os.path.isdir(p) else [os.path.basename(p)]:
                        algo_path = os.path.join(p, algo, "tuning") if os.path.exists(os.path.join(p, algo, "tuning")) else p
                        ts, env, run_path = find_latest_run_dir(algo_path)
                        if run_path:
                            try:
                                sweep = load_sweep_data(run_path)
                                for tid in target_tasks:
                                    all_task_data[tid]["runs"][algo] = sweep
                            except Exception as e:
                                print(f"Error loading {algo} from {algo_path}: {e}")

        # 3. For any remaining tasks with missing algorithms, auto-discover latest
        for task in TASKS:
            task_id = task["id"]
            policy = task["policy"]
            env = task["env"]

            discovered = discover_algorithm_sweeps(policy=policy, env_name=env, base_results_dir=base_results_dir)
            for algo, path in discovered.items():
                if algo not in all_task_data[task_id]["runs"]:
                    try:
                        all_task_data[task_id]["runs"][algo] = load_sweep_data(path)
                    except Exception as e:
                        print(f"[{task['name']}] Could not load {algo} from {path}: {e}")

        return all_task_data

    return (load_all_task_data,)


@app.cell
def _(
    ALGO_COLORS,
    ALGO_DISPLAY_NAMES,
    extract_best_configuration,
    np,
    os,
    plt,
):
    def plot_4task_grid(
        all_task_data,
        algo_list,
        title_prefix="Algorithms Comparison",
        metric_key="nn_weighted_VE",
        ylabel="Value Error (nn_weighted_VE)",
        log_scale=True,
        use_geom_mean=True,
        rank_by="final_window",
        rank_order="lower",
        window_size=200,
        save_path=None,
    ):
        """
        Plots a 2x2 grid comparing specified algorithms across the 4 core tasks.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False)
        axes = axes.flatten()

        task_keys = ["fixed_mountaincar", "fixed_fourrooms", "random_mountaincar", "random_fourrooms"]

        for idx, task_id in enumerate(task_keys):
            ax = axes[idx]
            if task_id not in all_task_data:
                continue

            task_obj = all_task_data[task_id]
            task_info = task_obj["task_info"]
            runs_dict = task_obj["runs"]

            ax.set_title(task_info["name"], fontsize=13, fontweight="bold", pad=8)
            plotted_any = False

            for algo in algo_list:
                if algo not in runs_dict:
                    continue

                sweep_data = runs_dict[algo]
                try:
                    seed_trajectories, best_label, best_idx, _ = extract_best_configuration(
                        sweep_data,
                        metric_key=metric_key,
                        rank_by=rank_by,
                        rank_order=rank_order,
                        window_size=window_size,
                    )
                except Exception:
                    continue

                n_seeds, time_steps = seed_trajectories.shape
                x = list(range(time_steps))
                color = ALGO_COLORS.get(algo, "#333333")
                display_name = ALGO_DISPLAY_NAMES.get(algo, algo)
                label = f"{display_name} ({best_label})" if best_label else display_name

                if use_geom_mean:
                    safe_arr = np.maximum(seed_trajectories, 1e-18)
                    log_arr = np.log(safe_arr)
                    log_mean = np.mean(log_arr, axis=0)
                    log_std = np.std(log_arr, axis=0)
                    geom_mean = np.exp(log_mean)
                    lower = np.exp(log_mean - log_std)
                    upper = np.exp(log_mean + log_std)

                    ax.plot(x, geom_mean, label=label, color=color, linewidth=2.0)
                    if n_seeds > 1:
                        ax.fill_between(x, lower, upper, color=color, alpha=0.18)
                else:
                    mean_curve = seed_trajectories.mean(axis=0)
                    std_curve = seed_trajectories.std(axis=0)
                    ax.plot(x, mean_curve, label=label, color=color, linewidth=2.0)
                    if n_seeds > 1:
                        lower_bound = np.maximum(mean_curve - std_curve, 1e-18) if log_scale else mean_curve - std_curve
                        ax.fill_between(x, lower_bound, mean_curve + std_curve, color=color, alpha=0.18)

                plotted_any = True

            if log_scale:
                ax.set_yscale("log")
            ax.set_xlabel("Update Steps", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            if plotted_any:
                ax.legend(loc="upper right", fontsize=8.5, frameon=True)
            else:
                ax.text(0.5, 0.5, "No Runs Found", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")

        fig.suptitle(f"{title_prefix} — {metric_key}", fontsize=15, fontweight="bold", y=0.995)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved 4-task grid plot to {save_path}")

        return fig

    return (plot_4task_grid,)


@app.cell
def _(
    ALGO_COLORS,
    ALGO_DISPLAY_NAMES,
    extract_best_configuration,
    np,
    os,
    plt,
):
    def plot_dual_metric_grid(
        all_task_data,
        algo_list,
        title="Sampled Algorithms: Value Error vs Greedy Policy Accuracy across 4 Tasks",
        metric_1="nn_weighted_VE",
        metric_2="nn_greedy_correct",
        rank_by="final_window",
        rank_order="lower",
        window_size=200,
        use_geom_mean=True,
        save_path=None,
    ):
        """
        Plots a 4x2 grid (4 rows = 4 tasks, 2 cols = metric_1 (log) & metric_2 (linear)).
        """
        task_keys = ["fixed_mountaincar", "fixed_fourrooms", "random_mountaincar", "random_fourrooms"]
        fig, axes = plt.subplots(4, 2, figsize=(16, 18), sharex=False)

        for row_idx, task_id in enumerate(task_keys):
            if task_id not in all_task_data:
                continue

            task_obj = all_task_data[task_id]
            task_info = task_obj["task_info"]
            runs_dict = task_obj["runs"]

            ax_m1 = axes[row_idx, 0]
            ax_m2 = axes[row_idx, 1]

            ax_m1.set_title(f"{task_info['name']} — {metric_1} (Log Scale)", fontsize=12, fontweight="bold")
            ax_m2.set_title(f"{task_info['name']} — {metric_2} (Linear [0, 1])", fontsize=12, fontweight="bold")

            plotted_m1 = False
            plotted_m2 = False

            for algo in algo_list:
                if algo not in runs_dict:
                    continue

                sweep_data = runs_dict[algo]
                color = ALGO_COLORS.get(algo, "#333333")
                display_name = ALGO_DISPLAY_NAMES.get(algo, algo)

                # Metric 1
                try:
                    m1_traj, best_label, _, _ = extract_best_configuration(
                        sweep_data, metric_key=metric_1, rank_by=rank_by, rank_order=rank_order, window_size=window_size
                    )
                    n_seeds, time_steps = m1_traj.shape
                    x = list(range(time_steps))
                    label = f"{display_name} ({best_label})" if best_label else display_name

                    if use_geom_mean:
                        safe_arr = np.maximum(m1_traj, 1e-18)
                        log_arr = np.log(safe_arr)
                        log_mean = np.mean(log_arr, axis=0)
                        log_std = np.std(log_arr, axis=0)
                        ax_m1.plot(x, np.exp(log_mean), label=label, color=color, linewidth=2.0)
                        if n_seeds > 1:
                            ax_m1.fill_between(x, np.exp(log_mean - log_std), np.exp(log_mean + log_std), color=color, alpha=0.18)
                    else:
                        mean_curve = m1_traj.mean(axis=0)
                        std_curve = m1_traj.std(axis=0)
                        ax_m1.plot(x, mean_curve, label=label, color=color, linewidth=2.0)
                        if n_seeds > 1:
                            ax_m1.fill_between(x, np.maximum(mean_curve - std_curve, 1e-18), mean_curve + std_curve, color=color, alpha=0.18)

                    plotted_m1 = True
                except Exception:
                    pass

                # Metric 2
                try:
                    m2_traj, _, _, _ = extract_best_configuration(
                        sweep_data, metric_key=metric_2, rank_by=rank_by, rank_order=rank_order, window_size=window_size
                    )
                    n_seeds, time_steps = m2_traj.shape
                    x = list(range(time_steps))
                    mean_curve2 = m2_traj.mean(axis=0)
                    std_curve2 = m2_traj.std(axis=0)
                    ax_m2.plot(x, mean_curve2, label=label, color=color, linewidth=2.0)
                    if n_seeds > 1:
                        ax_m2.fill_between(x, np.clip(mean_curve2 - std_curve2, 0, 1), np.clip(mean_curve2 + std_curve2, 0, 1), color=color, alpha=0.18)
                    plotted_m2 = True
                except Exception:
                    pass

            ax_m1.set_yscale("log")
            ax_m1.set_xlabel("Update Steps", fontsize=10)
            ax_m1.set_ylabel(metric_1, fontsize=10)
            ax_m1.grid(True, which="both", linestyle="--", alpha=0.5)
            if plotted_m1:
                ax_m1.legend(loc="upper right", fontsize=8, frameon=True)

            ax_m2.set_xlabel("Update Steps", fontsize=10)
            ax_m2.set_ylabel(metric_2, fontsize=10)
            ax_m2.set_ylim(-0.02, 1.02)
            ax_m2.grid(True, which="both", linestyle="--", alpha=0.5)
            if plotted_m2:
                ax_m2.legend(loc="lower right", fontsize=8, frameon=True)

        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved dual-metric grid plot to {save_path}")

        return fig

    return


@app.cell
def _(ALGO_DISPLAY_NAMES, extract_best_configuration, os, pd):
    def generate_task_summary_table(
        all_task_data,
        algo_list,
        metric_key="nn_weighted_VE",
        window_size=200,
        rank_by="final_window",
        rank_order="lower",
        save_path=None,
    ):
        """
        Creates a unified summary table across all 4 tasks for the specified algorithms.
        """
        task_keys = ["fixed_mountaincar", "fixed_fourrooms", "random_mountaincar", "random_fourrooms"]
        rows = []

        for task_id in task_keys:
            if task_id not in all_task_data:
                continue

            task_obj = all_task_data[task_id]
            task_info = task_obj["task_info"]
            runs_dict = task_obj["runs"]

            for algo in algo_list:
                if algo not in runs_dict:
                    continue

                sweep_data = runs_dict[algo]
                try:
                    seed_trajectories, best_label, best_idx, _ = extract_best_configuration(
                        sweep_data,
                        metric_key=metric_key,
                        rank_by=rank_by,
                        rank_order=rank_order,
                        window_size=window_size,
                    )
                    n_seeds, time_steps = seed_trajectories.shape
                    win = max(1, min(time_steps, window_size))

                    auc_val = float(seed_trajectories.mean())
                    window_vals = seed_trajectories[:, -win:]
                    window_mean = float(window_vals.mean())
                    window_std = float(window_vals.mean(axis=-1).std()) if n_seeds > 1 else 0.0

                    final_vals = seed_trajectories[:, -1]
                    final_mean = float(final_vals.mean())
                    final_std = float(final_vals.std()) if n_seeds > 1 else 0.0

                    min_val = float(seed_trajectories.min())
                    max_val = float(seed_trajectories.max())

                    display_name = ALGO_DISPLAY_NAMES.get(algo, algo)

                    rows.append({
                        "Task": task_info["name"],
                        "Algorithm": display_name,
                        "Best Hyperparameters": best_label,
                        f"Converged Window (Past {win} steps)": f"{window_mean:.4e} ± {window_std:.2e}",
                        f"AUC ({metric_key})": f"{auc_val:.4e}",
                        f"Final ({metric_key})": f"{final_mean:.4e} ± {final_std:.2e}",
                        f"Min {metric_key}": f"{min_val:.4e}",
                        f"Max {metric_key}": f"{max_val:.4e}",
                        "Seeds": int(n_seeds),
                        "Total Steps": int(time_steps),
                        "Run Directory": sweep_data.get("run_dir", "N/A"),
                    })
                except Exception as e:
                    print(f"Error summarizing {algo} on {task_info['name']}: {e}")

        summary_df = pd.DataFrame(rows)

        if save_path and not summary_df.empty:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            if save_path.endswith(".csv"):
                summary_df.to_csv(save_path, index=False)
            elif save_path.endswith(".json"):
                summary_df.to_json(save_path, orient="records", indent=4)
            print(f"Summary table saved to {save_path}")

        return summary_df

    return (generate_task_summary_table,)


@app.cell
def _(mo):
    mo.md("""
    # 🔬 Reinforcement Learning Sweep Benchmark Suite
    ### Cross-Algorithm & Cross-Task Learning Curves & Converged Metric Analysis
    Tasks evaluated:
    1. **Fixed Policy** — MountainCar-v0
    2. **Fixed Policy** — FourRooms-misc
    3. **Random Policy** — MountainCar-v0
    4. **Random Policy** — FourRooms-misc
    """)
    return


@app.cell
def _():
    # Paste your sweep directories here to save them permanently in code/git:
    SWEEPS_TO_LOAD = [
        "results/fixed/sweeps/fixed_MountainCar-v0_20260828_094649",
        "results/fixed/sweeps/fixed_MountainCar-v0_20260828_093547",
        "results/fixed/sweeps/fixed_FourRooms-misc_20260828_093737",
        "results/fixed/sweeps/fixed_FourRooms-misc_20260828_093104",

        "results/random/sweeps/random_FourRooms-misc_20260828_093356",
        "results/random/sweeps/random_FourRooms-misc_20260828_092803",
        "results/random/sweeps/random_MountainCar-v0_20260828_093356",
        "results/random/sweeps/random_MountainCar-v0_20260828_093944",
    ]
    return (SWEEPS_TO_LOAD,)


@app.cell
def _(mo):
    base_dir_input = mo.ui.text(value="results", label="Base Results Dir")
    window_size_slider = mo.ui.slider(start=10, stop=500, step=10, value=200, label="Tail Window Size")
    rank_by_dropdown = mo.ui.dropdown(
        options=["final_window", "auc", "final_step", "min"],
        value="final_window",
        label="Rank Configs By",
    )
    use_geom_mean_checkbox = mo.ui.checkbox(value=True, label="Geometric Mean Bands")

    controls = mo.hstack([base_dir_input, window_size_slider, rank_by_dropdown, use_geom_mean_checkbox], justify="start")
    controls
    return (
        base_dir_input,
        rank_by_dropdown,
        use_geom_mean_checkbox,
        window_size_slider,
    )


@app.cell
def _(SWEEPS_TO_LOAD, base_dir_input, load_all_task_data, mo):
    task_data = load_all_task_data(
        base_results_dir=base_dir_input.value,
        custom_batches=SWEEPS_TO_LOAD if SWEEPS_TO_LOAD else None,
    )
    total_loaded = sum(len(v["runs"]) for v in task_data.values())
    mo.md(f"✅ **Sweep Runs Loaded:** Loaded **{total_loaded}** algorithm sweep runs across 4 tasks.")
    return (task_data,)


@app.cell
def _(
    SAMPLED_ALGOS,
    mo,
    plot_4task_grid,
    rank_by_dropdown,
    task_data,
    use_geom_mean_checkbox,
    window_size_slider,
):
    fig_sampled = plot_4task_grid(
        task_data,
        algo_list=SAMPLED_ALGOS,
        title_prefix="Sampled Algorithms Performance Across 4 Tasks",
        metric_key="nn_weighted_VE",
        ylabel="Value Error (nn_weighted_VE)",
        log_scale=True,
        use_geom_mean=use_geom_mean_checkbox.value,
        rank_by=rank_by_dropdown.value,
        window_size=window_size_slider.value,
        save_path="results/comparison_sampled_4tasks_VE.png",
    )
    mo.vstack([
        mo.md("## 📊 1. Sampled Algorithms: 4-Task Value Error (`nn_weighted_VE`)"),
        fig_sampled,
    ])
    return


@app.cell
def _(
    EXACT_ALGOS,
    mo,
    plot_4task_grid,
    rank_by_dropdown,
    task_data,
    use_geom_mean_checkbox,
    window_size_slider,
):
    fig_exact = plot_4task_grid(
        task_data,
        algo_list=EXACT_ALGOS,
        title_prefix="Exact Algorithms Performance Across 4 Tasks",
        metric_key="nn_weighted_VE",
        ylabel="Value Error (nn_weighted_VE)",
        log_scale=True,
        use_geom_mean=use_geom_mean_checkbox.value,
        rank_by=rank_by_dropdown.value,
        window_size=window_size_slider.value,
        save_path="results/comparison_exact_4tasks_VE.png",
    )
    mo.vstack([
        mo.md("## 📊 2. Exact Algorithms: 4-Task Value Error (`nn_weighted_VE`)"),
        fig_exact,
    ])
    return


@app.cell
def _(
    EXACT_ALGOS,
    plot_4task_grid,
    rank_by_dropdown,
    task_data,
    use_geom_mean_checkbox,
    window_size_slider,
):
    fig_greedy_exact = plot_4task_grid(
        task_data,
        algo_list=EXACT_ALGOS,
        title_prefix="Exact Algorithms Performance Across 4 Tasks",
        metric_key="nn_greedy_correct",
        ylabel="Greedy Policy Accuracy",
        log_scale=False,
        use_geom_mean=use_geom_mean_checkbox.value,
        rank_by=rank_by_dropdown.value,
        window_size=window_size_slider.value,
        save_path="results/comparison_exact_4tasks_greedy_acc.png",
    )
    fig_greedy_exact
    return


@app.cell
def _(
    SAMPLED_ALGOS,
    plot_4task_grid,
    rank_by_dropdown,
    task_data,
    use_geom_mean_checkbox,
    window_size_slider,
):
    fig_greedy_sampled = plot_4task_grid(
        task_data,
        algo_list=SAMPLED_ALGOS,
        title_prefix="Exact Algorithms Performance Across 4 Tasks",
        metric_key="nn_greedy_correct",
        ylabel="Greedy Policy Accuracy",
        log_scale=False,
        use_geom_mean=use_geom_mean_checkbox.value,
        rank_by=rank_by_dropdown.value,
        window_size=window_size_slider.value,
        save_path="results/comparison_sampled_4tasks_greedy_acc.png",
    )
    fig_greedy_sampled
    return


@app.cell
def _(
    EXACT_ALGOS,
    SAMPLED_ALGOS,
    generate_task_summary_table,
    mo,
    rank_by_dropdown,
    task_data,
    window_size_slider,
):
    sampled_table = generate_task_summary_table(
        task_data,
        algo_list=SAMPLED_ALGOS,
        metric_key="nn_weighted_VE",
        window_size=window_size_slider.value,
        rank_by=rank_by_dropdown.value,
        save_path="results/sampled_algorithms_converged_summary.csv",
    )

    exact_table = generate_task_summary_table(
        task_data,
        algo_list=EXACT_ALGOS,
        metric_key="nn_weighted_VE",
        window_size=window_size_slider.value,
        rank_by=rank_by_dropdown.value,
        save_path="results/exact_algorithms_converged_summary.csv",
    )

    mo.vstack([
        mo.md(f"## 📋 4. Sampled Algorithms: Converged Summary Table (Tail Window: Past {window_size_slider.value} steps)"),
        mo.ui.table(sampled_table),
        mo.md(f"## 📋 5. Exact Algorithms: Converged Summary Table (Tail Window: Past {window_size_slider.value} steps)"),
        mo.ui.table(exact_table),
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
