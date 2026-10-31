import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ENV_DISPLAY_NAMES = {
    "FourRooms-misc": "FourRooms (Sparse)",
    "fourrooms-dense": "FourRooms (Dense)",
    "fourrooms-mines": "FourRooms + Mines",
    "fourrooms-mines-dense": "FourRooms + Mines (Dense)",
    "eightrooms-misc": "EightRooms (Sparse)",
    "eightrooms-dense": "EightRooms (Dense)",
    "whirlpool-misc": "Whirlpool",
    "MountainCar-v0": "MountainCar (Sparse)",
    "mountaincar-dense": "MountainCar (Dense)",
    "SpaceInvadersExactValue": "Space Invaders",
}

ENV_COLORS = {
    "FourRooms-misc": "#1f77b4",
    "fourrooms-dense": "#aec7e8",
    "fourrooms-mines": "#d62728",
    "fourrooms-mines-dense": "#ff9896",
    "eightrooms-misc": "#2ca02c",
    "eightrooms-dense": "#98df8a",
    "whirlpool-misc": "#9467bd",
    "MountainCar-v0": "#ff7f0e",
    "mountaincar-dense": "#ffbb78",
    "SpaceInvadersExactValue": "#8c564b",
}

def load_algo_performance(results_dir: str, env_name: str, algo_name: str, lr_multipliers=["0.5x", "1.0x", "2.0x"]):
    """Loads best performance for an algorithm from summary_metrics.csv or results.npz."""
    csv_path = os.path.join(results_dir, "summary_metrics.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            algo_map = {
                "td0": ["TD0", "td0", "td_0"],
                "e0": ["E0", "e0", "E_0"],
                "mc": ["MC", "mc"],
                "td_lambda": ["TD_lambda", "td_lambda"],
            }
            valid_names = algo_map.get(algo_name.lower(), [algo_name])
            sub = df[(df["environment"] == env_name) & (df["algorithm"].isin(valid_names))]
            if len(sub) > 0:
                return float(sub.iloc[0]["final_score"])
        except Exception:
            pass

    npz_path = os.path.join(results_dir, env_name, "results.npz")
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)
    metric_candidates = ["V_start", "returned_episode_returns", "mean_rew"]
    chosen_metric = None

    for m in metric_candidates:
        if any(f"{algo_name}_{mult}_{m}" in data for mult in lr_multipliers):
            chosen_metric = m
            break

    if chosen_metric is None:
        return None

    best_score = -float("inf")
    for mult in lr_multipliers:
        key = f"{algo_name}_{mult}_{chosen_metric}"
        if key in data:
            arr = data[key]
            # Use final 10 updates
            score = float(np.mean(arr[:, -10:]))
            if score > best_score:
                best_score = score

    return best_score if best_score != -float("inf") else None


def main():
    parser = argparse.ArgumentParser(description="Generate 4-way environment diagnostic scatter plots.")
    parser.add_argument("--env_props_csv", type=str, default="results/env_properties.csv", help="Path to env properties CSV")
    parser.add_argument("--expected_dir", type=str, default="results/expected_algo_sweep", help="Directory of expected sweeps")
    parser.add_argument("--sampled_dir", type=str, default="results/sampled_algo_sweep", help="Directory of sampled sweeps")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.env_props_csv):
        print(f"Warning: {args.env_props_csv} not found. Running analyze_env_properties.py first...")
        from scripts.analyze_env_properties import ALL_ENVS, analyze_env
        all_rows = [analyze_env(env, gt_updates=80) for env in ALL_ENVS]
        df_props = pd.DataFrame(all_rows)
        df_props.to_csv(args.env_props_csv, index=False)
    else:
        df_props = pd.read_csv(args.env_props_csv)

    print(f"Loaded {len(df_props)} environment profiles.")

    # Match performance metrics
    records = []
    for _, row in df_props.iterrows():
        env = row["env_name"]
        exp_td = load_algo_performance(args.expected_dir, env, "td0")
        exp_e = load_algo_performance(args.expected_dir, env, "e0")

        samp_td = load_algo_performance(args.sampled_dir, env, "td0")
        samp_e = load_algo_performance(args.sampled_dir, env, "e0")

        # Compute TD / E ratios
        exp_ratio = None
        if exp_td is not None and exp_e is not None:
            # Safe ratio handling for negative or zero values:
            if exp_e != 0:
                exp_ratio = exp_td / exp_e
            else:
                exp_ratio = 1.0

        samp_ratio = None
        if samp_td is not None and samp_e is not None:
            if samp_e != 0:
                samp_ratio = samp_td / samp_e
            else:
                samp_ratio = 1.0

        rec = row.to_dict()
        rec["exp_td"] = exp_td
        rec["exp_e"] = exp_e
        rec["exp_ratio"] = exp_ratio
        rec["samp_td"] = samp_td
        rec["samp_e"] = samp_e
        rec["samp_ratio"] = samp_ratio
        records.append(rec)

    df = pd.DataFrame(records)

    # 4-panel diagnostic scatter layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150)
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # Helper function to plot points and annotations
    def plot_scatter(ax, x_key, y_key, x_label, y_label, title, is_ratio_y=True, log_x=False, log_y=False):
        valid = df.dropna(subset=[x_key, y_key])
        if len(valid) == 0:
            ax.text(0.5, 0.5, f"Pending Sweep Data\n({x_label} vs {y_label})", ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.grid(True, alpha=0.3)
            return

        for _, r in valid.iterrows():
            env = r["env_name"]
            x = r[x_key]
            y = r[y_key]
            color = ENV_COLORS.get(env, "#333333")
            display_name = ENV_DISPLAY_NAMES.get(env, env)

            ax.scatter(x, y, color=color, s=110, edgecolors="black", linewidth=1.2, zorder=4)
            # Offset annotation
            ax.annotate(
                display_name,
                (x, y),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="medium",
                alpha=0.9,
                zorder=5,
            )

        if is_ratio_y:
            ax.axhline(1.0, color="gray", linestyle="--", lw=1.2, alpha=0.7, label="TD = E Parity")

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, alpha=0.3)

    # Plot 1: Asymmetry
    # X: ||K||_2 / ||S||_2 | Y: Expected TD / Expected E
    plot_scatter(
        ax1,
        x_key="asymmetry_ratio",
        y_key="exp_ratio",
        x_label=r"Transition Asymmetry $\frac{\|K\|_2}{\|S\|_2}$ (under trained $\pi$)",
        y_label=r"Expected Performance Ratio $\frac{\text{Perf}(TD)}{\text{Perf}(E)}$",
        title=r"[Plot 1: Asymmetry] Asymmetry vs Relative TD Advantage",
    )

    # Plot 2: Sparsity
    # X: Reward Density / Entropy | Y: Expected TD / Expected E
    plot_scatter(
        ax2,
        x_key="reward_density",
        y_key="exp_ratio",
        x_label=r"Reward Density $\sum_{s: |R(s)|>0} \mu(s)$ (visited non-zero reward)",
        y_label=r"Expected Performance Ratio $\frac{\text{Perf}(TD)}{\text{Perf}(E)}$",
        title=r"[Plot 2: Sparsity] Reward Density vs Relative TD Advantage",
    )

    # Plot 3: Sample Variance Penalty
    # X: Terminal Flux mu(terminals) | Y: Var(Grad E) / Var(Grad TD)
    plot_scatter(
        ax3,
        x_key="terminal_flux",
        y_key="grad_var_ratio",
        x_label=r"Terminal Flux $\mu(\text{terminals})$",
        y_label=r"Sample Gradient Variance Ratio $\frac{\text{Var}(\nabla \mathcal{L}_E)}{\text{Var}(\nabla \mathcal{L}_{TD})}$",
        title=r"[Plot 3: Variance Penalty] Terminal Boundary Flux vs Sample Gradient Noise",
        is_ratio_y=False,
    )

    # Plot 4: Sampled Transfer
    # X: Expected (TD/E) ratio | Y: Sampled (TD/E) ratio
    plot_scatter(
        ax4,
        x_key="exp_ratio",
        y_key="samp_ratio",
        x_label=r"Expected Performance Ratio $\frac{\text{Perf}(TD)}{\text{Perf}(E)}$",
        y_label=r"Sampled Performance Ratio $\frac{\text{Perf}(TD)}{\text{Perf}(E)}$",
        title=r"[Plot 4: Sampled Transfer] Expected vs Sampled Relative Performance",
        is_ratio_y=True,
    )
    # Add diagonal y=x line on plot 4
    ax4.axline((1, 1), slope=1, color="blue", linestyle=":", lw=1.2, alpha=0.6, label="Perfect Transfer (y=x)")

    # Unified Legend below
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, markeredgecolor="black", label=n)
        for env_id, n in ENV_DISPLAY_NAMES.items()
        for c in [ENV_COLORS.get(env_id, "#333")]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Reinforcement Learning Diagnostic Dashboard: Understanding E vs TD Performance Across Environments", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    pdf_out = os.path.join(args.out_dir, "diagnostic_env_scatters.pdf")
    png_out = os.path.join(args.out_dir, "diagnostic_env_scatters.png")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.savefig(png_out, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"Diagnostic dashboard generated successfully:")
    print(f"  PDF: {pdf_out}")
    print(f"  PNG: {png_out}")

if __name__ == "__main__":
    main()
