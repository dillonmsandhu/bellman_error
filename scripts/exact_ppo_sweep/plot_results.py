"""
scripts/exact_ppo_sweep/plot_results.py

Generates publication-quality charts for the Exact PPO Evaluation Sweep:
1. All-Algorithms Charts (including E(lambda) and TD(lambda)):
   - returns_learning_curves_all_algos.pdf / .png:
     Raw expected policy return V^pi(s_0) across all 5 algorithms:
     MC, TD(0), E(0), TD(λ=0.8), E(λ=0.8).
   - standardized_performance_learning_curves_all_algos.pdf / .png:
     Standardized performance (V_start - V_rand) / (V* - V_rand) across all 5 algorithms.
   - final_standardized_performance_bars_all_algos.pdf / .png:
     Grouped bar charts of final standardized performance across all 5 algorithms.
   - auc_standardized_performance_bars_all_algos.pdf / .png:
     Grouped bar charts of AUC standardized performance across all 5 algorithms.

2. Triad Charts (Exact TD, Exact E Minimization, Exact MC):
   - returns_learning_curves.pdf / .png:
     Raw expected policy return V^pi(s_0) across the 3 core algorithms.
   - standardized_performance_learning_curves.pdf / .png:
     Standardized performance across the 3 core algorithms.
   - final_standardized_performance_bars.pdf / .png:
     Grouped bar charts of final standardized performance across the 3 core algorithms.
   - auc_standardized_performance_bars.pdf / .png:
     Grouped bar charts of AUC standardized performance across the 3 core algorithms.
"""

from __future__ import annotations
import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Publication font formatting
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
matplotlib.rcParams["axes.edgecolor"] = "#333333"
matplotlib.rcParams["axes.linewidth"] = 0.8

ALL_ENVS = [
    "mountaincar",
    "FourRooms-misc",
    "fourrooms-mines",
    "fourrooms-mines-dense",
    "eightrooms-dense",
    "eightrooms-misc",
    "whirlpool-misc",
    "SpaceInvadersExactValue",
    "mountaincar-dense",
]

# Color palettes
ALGO_COLORS = {
    "mc": "#1f77b4",          # Steel Blue
    "td0": "#ff7f0e",         # Dark Orange
    "e0": "#2ca02c",          # Forest Green
    "td_lambda": "#d62728",   # Crimson Red
    "e_lambda": "#9467bd",    # Purple
}

# Labels for all algorithms comparison
ALGOS_ALL = ["mc", "td0", "e0", "td_lambda", "e_lambda"]
ALGO_LABELS_ALL = {
    "mc": "Exact MC",
    "td0": "Exact TD(0)",
    "e0": "Exact E(0)",
    "td_lambda": "Exact TD(λ=0.8)",
    "e_lambda": "Exact E(λ=0.8)",
}

# Labels for 3-algorithm triad comparison
ALGOS_TRIAD = ["td0", "e0", "mc"]
ALGO_LABELS_TRIAD = {
    "td0": "Exact TD",
    "e0": "Exact E Minimization",
    "mc": "Exact MC",
}

ENV_TITLES = {
    "mountaincar": "Mountain Car (Sparse)",
    "FourRooms-misc": "FourRooms (Sparse)",
    "fourrooms-mines": "FourRooms + Mines",
    "fourrooms-mines-dense": "FourRooms + Mines (Dense)",
    "eightrooms-dense": "EightRooms (Dense)",
    "eightrooms-misc": "EightRooms (Sparse)",
    "whirlpool-misc": "Whirlpool (Irreversible)",
    "SpaceInvadersExactValue": "Space Invaders",
    "mountaincar-dense": "Mountain Car (Dense)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Exact PPO Sweep Publication Plots")
    parser.add_argument("--results_dir", type=str, default="results/exact_ppo_sweep", help="Results directory root")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory (default: same as results_dir/plots)")
    parser.add_argument("--ci", type=str, default="95", choices=["95", "sem"], help="Confidence interval band: '95' (95% CI via Student-t) or 'sem' (+/-1 SEM)")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "all", "triad"], help="Which plots to generate: 'all' (all 5 algos), 'triad' (3 algos), or 'both'")
    return parser.parse_args()


def load_all_env_data(results_dir: str):
    """Loads all out.pkl checkpoints and summary.csv tables from results_dir."""
    env_data = {}
    summary_dfs = []

    for env_name in ALL_ENVS:
        env_dir = os.path.join(results_dir, env_name)
        pkl_path = os.path.join(env_dir, "out.pkl")
        csv_path = os.path.join(env_dir, "summary.csv")

        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                env_data[env_name] = data
            except Exception as e:
                print(f"[Warning] Failed to load {pkl_path}: {e}")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                summary_dfs.append(df)
            except Exception as e:
                print(f"[Warning] Failed to read {csv_path}: {e}")

    combined_summary = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()
    return env_data, combined_summary


def compute_ci_band(data_2d: np.ndarray, ci_type: str = "95") -> tuple[np.ndarray, np.ndarray]:
    """
    Computes mean curve and confidence interval half-width across random seeds (axis 0).
    - ci_type="95": 95% Confidence Interval using Student's t distribution.
    - ci_type="sem": +/- 1 Standard Error of the Mean (SEM).
    """
    n_seeds = data_2d.shape[0]
    mean_curve = np.mean(data_2d, axis=0)
    if n_seeds > 1:
        sem_curve = np.std(data_2d, axis=0, ddof=1) / np.sqrt(n_seeds)
        if ci_type == "95":
            try:
                import scipy.stats as stats
                t_crit = float(stats.t.ppf(0.975, df=n_seeds - 1))
            except Exception:
                t_crit = 1.96
            err_curve = t_crit * sem_curve
        elif ci_type == "sem":
            err_curve = sem_curve
        else:
            raise ValueError(f"Unknown ci_type: {ci_type}")
    else:
        err_curve = np.zeros_like(mean_curve)
    return mean_curve, err_curve


def plot_learning_curves(
    env_data: dict,
    out_pdf_path: str,
    algos_ordered: list[str],
    algo_labels: dict[str, str],
    standardized: bool = False,
    ci_type: str = "95",
):
    """
    Generates 3x3 grid of learning curves across PPO updates for the specified algorithms.
    If standardized=True, plots (V_start - V_rand) / (V* - V_rand).
    If standardized=False, plots raw return V^pi(s_0).
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    legend_handles = {}

    for idx, env_name in enumerate(ALL_ENVS):
        ax = axes[idx]
        title = ENV_TITLES.get(env_name, env_name)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.grid(True, linestyle="--", alpha=0.35)

        if env_name not in env_data or "runs" not in env_data[env_name]:
            ax.text(0.5, 0.5, "Data Not Found", ha="center", va="center", transform=ax.transAxes, color="#888888")
            continue

        baselines = env_data[env_name].get("baselines", {})
        v_star = baselines.get("v_star")
        v_rand = baselines.get("v_rand")

        # Baseline horizontal reference lines
        if standardized:
            ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)
        else:
            if v_star is not None:
                ax.axhline(v_star, color="#222222", linestyle=":", linewidth=1.2, alpha=0.7)
            if v_rand is not None:
                ax.axhline(v_rand, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)

        runs = env_data[env_name]["runs"]

        for algo_key in algos_ordered:
            if algo_key not in runs:
                continue
            run_dict = runs[algo_key]

            if standardized:
                traj = run_dict.get("standardized_v_start")
                if traj is None and "V_start" in run_dict and v_star is not None and v_rand is not None:
                    delta = baselines.get("delta", v_star - v_rand)
                    traj = (run_dict["V_start"] - v_rand) / (delta + 1e-12)
            else:
                traj = run_dict.get("V_start")

            if traj is None:
                continue

            mean_curve, err_curve = compute_ci_band(traj, ci_type=ci_type)
            steps = np.arange(1, len(mean_curve) + 1)

            color = ALGO_COLORS.get(algo_key, "#333333")
            label = algo_labels.get(algo_key, algo_key)

            line = ax.plot(steps, mean_curve, color=color, linewidth=1.9, label=label)[0]
            if np.any(err_curve > 0):
                ax.fill_between(steps, mean_curve - err_curve, mean_curve + err_curve, color=color, alpha=0.18)

            if algo_key not in legend_handles:
                legend_handles[algo_key] = line

        ax.set_xlabel("PPO Updates", fontsize=10)
        ylabel = "Standardized Return" if standardized else "Expected Return $V^{\\pi}(s_0)$"
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)

    # Reference line dummy handles for global legend
    ref_optimal = Line2D([0], [0], color="#222222", linestyle=":", linewidth=1.4, label="Optimal $V^*$")
    ref_random = Line2D([0], [0], color="#888888", linestyle=":", linewidth=1.2, label="Random $V^{\\pi_{\\mathrm{rand}}}$")

    # Global legend at the top
    handles = [legend_handles[k] for k in algos_ordered if k in legend_handles]
    labels = [algo_labels[k] for k in algos_ordered if k in legend_handles]
    handles.extend([ref_optimal, ref_random])
    labels.extend(["Optimal $V^*$", "Random $V^{\\pi_{\\mathrm{rand}}}$"])

    ncol = min(len(handles), 7)
    fontsize = 10 if len(handles) > 5 else 11

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=ncol,
        frameon=True,
        fontsize=fontsize,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_pdf_path, dpi=300)
    png_path = os.path.splitext(out_pdf_path)[0] + ".png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"[Plot Generated] {out_pdf_path}")
    print(f"[Plot Generated] {png_path}")


def plot_performance_bars(
    summary_df: pd.DataFrame,
    metric_col: str,
    sem_col: str,
    title: str,
    ylabel: str,
    out_pdf_path: str,
    algos_ordered: list[str],
    algo_labels: dict[str, str],
):
    """Generates grouped bar charts comparing performance across environments."""
    if summary_df.empty:
        return

    envs_present = [e for e in ALL_ENVS if e in summary_df["env"].unique()]
    algos_present = [a for a in algos_ordered if a in summary_df["algo"].unique()]

    if not envs_present or not algos_present:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    n_envs = len(envs_present)
    n_algos = len(algos_present)
    bar_width = 0.8 / n_algos
    x = np.arange(n_envs)

    ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)

    for i, algo in enumerate(algos_present):
        means = []
        sems = []
        for e in envs_present:
            sub = summary_df[(summary_df["env"] == e) & (summary_df["algo"] == algo)]
            if not sub.empty:
                means.append(sub[metric_col].values[0])
                sems.append(sub[sem_col].values[0] if sem_col in sub else 0.0)
            else:
                means.append(0.0)
                sems.append(0.0)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            yerr=sems,
            width=bar_width,
            label=algo_labels.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#333333"),
            edgecolor="#222222",
            linewidth=0.8,
            capsize=3,
            alpha=0.9,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_TITLES.get(e, e) for e in envs_present], rotation=30, ha="right", fontsize=9.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=10 if len(algos_present) > 3 else 10.5, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_pdf_path, dpi=300)
    png_path = os.path.splitext(out_pdf_path)[0] + ".png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"[Plot Generated] {out_pdf_path}")
    print(f"[Plot Generated] {png_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else os.path.join(results_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("Exact PPO Sweep Plot Generation")
    print(f"Results Directory: {results_dir}")
    print(f"Save Directory:    {save_dir}")
    print(f"Confidence Band:   {args.ci} ({'95% CI' if args.ci == '95' else '+/-1 SEM'})")
    print(f"Mode:              {args.mode}")
    print("=" * 70)

    env_data, summary_df = load_all_env_data(results_dir)
    print(f"Loaded data for {len(env_data)} environments.")

    # -------------------------------------------------------------
    # 1. All Algorithms Plots (MC, TD(0), E(0), TD(λ), E(λ))
    # -------------------------------------------------------------
    if args.mode in ("all", "both"):
        print("\n--- Generating All-Algorithms Plots (5 algos, including E(λ) & TD(λ)) ---")
        # Raw Returns
        plot_learning_curves(
            env_data=env_data,
            out_pdf_path=os.path.join(save_dir, "returns_learning_curves_all_algos.pdf"),
            algos_ordered=ALGOS_ALL,
            algo_labels=ALGO_LABELS_ALL,
            standardized=False,
            ci_type=args.ci,
        )
        # Standardized Returns / Performance
        plot_learning_curves(
            env_data=env_data,
            out_pdf_path=os.path.join(save_dir, "standardized_performance_learning_curves_all_algos.pdf"),
            algos_ordered=ALGOS_ALL,
            algo_labels=ALGO_LABELS_ALL,
            standardized=True,
            ci_type=args.ci,
        )
        plot_learning_curves(
            env_data=env_data,
            out_pdf_path=os.path.join(save_dir, "standardized_returns_learning_curves_all_algos.pdf"),
            algos_ordered=ALGOS_ALL,
            algo_labels=ALGO_LABELS_ALL,
            standardized=True,
            ci_type=args.ci,
        )
        # Bar charts
        plot_performance_bars(
            summary_df=summary_df,
            metric_col="std_perf_final_mean",
            sem_col="std_perf_final_sem",
            title="Final Standardized Policy Performance across All Algorithms",
            ylabel="Standardized Performance (Random=0, Optimal=1)",
            out_pdf_path=os.path.join(save_dir, "final_standardized_performance_bars_all_algos.pdf"),
            algos_ordered=ALGOS_ALL,
            algo_labels=ALGO_LABELS_ALL,
        )
        plot_performance_bars(
            summary_df=summary_df,
            metric_col="std_perf_auc_mean",
            sem_col="std_perf_auc_sem",
            title="AUC Standardized Policy Performance across All Algorithms",
            ylabel="AUC Standardized Performance",
            out_pdf_path=os.path.join(save_dir, "auc_standardized_performance_bars_all_algos.pdf"),
            algos_ordered=ALGOS_ALL,
            algo_labels=ALGO_LABELS_ALL,
        )

    # -------------------------------------------------------------
    # 2. Triad Plots (Exact TD, Exact E Minimization, Exact MC)
    # -------------------------------------------------------------
    if args.mode in ("triad", "both"):
        print("\n--- Generating Triad Plots (Exact TD, Exact E Minimization, Exact MC) ---")
        # Raw Returns
        plot_learning_curves(
            env_data=env_data,
            out_pdf_path=os.path.join(save_dir, "returns_learning_curves.pdf"),
            algos_ordered=ALGOS_TRIAD,
            algo_labels=ALGO_LABELS_TRIAD,
            standardized=False,
            ci_type=args.ci,
        )
        # Standardized Returns / Performance
        plot_learning_curves(
            env_data=env_data,
            out_pdf_path=os.path.join(save_dir, "standardized_performance_learning_curves.pdf"),
            algos_ordered=ALGOS_TRIAD,
            algo_labels=ALGO_LABELS_TRIAD,
            standardized=True,
            ci_type=args.ci,
        )
        plot_learning_curves(
            env_data=env_data,
            out_pdf_path=os.path.join(save_dir, "standardized_returns_learning_curves.pdf"),
            algos_ordered=ALGOS_TRIAD,
            algo_labels=ALGO_LABELS_TRIAD,
            standardized=True,
            ci_type=args.ci,
        )
        # Bar charts
        plot_performance_bars(
            summary_df=summary_df,
            metric_col="std_perf_final_mean",
            sem_col="std_perf_final_sem",
            title="Final Standardized Policy Performance (PPO Exact Evaluator)",
            ylabel="Standardized Performance (Random=0, Optimal=1)",
            out_pdf_path=os.path.join(save_dir, "final_standardized_performance_bars.pdf"),
            algos_ordered=ALGOS_TRIAD,
            algo_labels=ALGO_LABELS_TRIAD,
        )
        plot_performance_bars(
            summary_df=summary_df,
            metric_col="std_perf_auc_mean",
            sem_col="std_perf_auc_sem",
            title="Area Under Curve (AUC) Standardized Policy Performance (PPO Exact Evaluator)",
            ylabel="AUC Standardized Performance",
            out_pdf_path=os.path.join(save_dir, "auc_standardized_performance_bars.pdf"),
            algos_ordered=ALGOS_TRIAD,
            algo_labels=ALGO_LABELS_TRIAD,
        )

    print("\n[Complete] All requested PPO publication plots successfully generated.")


if __name__ == "__main__":
    main()


