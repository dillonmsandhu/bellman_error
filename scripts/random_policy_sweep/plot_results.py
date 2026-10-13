"""
scripts/random_policy_sweep/plot_results.py

Consolidated comparison plots for Random Policy Evaluation Benchmark:
Algorithms: Sampled MC, TD(0), TD(lambda), and E(0) (Oct 8).

Visuals generated (strictly PDF):
1. VE during learning:
   - Line chart across environments: ve_lines_across_envs.pdf
   - Bar chart across environments: ve_bars_across_envs.pdf
2. Advantage Cosine Sim during learning:
   - Line chart across environments: advantage_cossim_lines_across_envs.pdf
   - Bar chart across environments: advantage_cossim_bars_across_envs.pdf
3. V_start of the greedy policy during learning:
   - Line chart across environments: v_start_greedy_lines_across_envs.pdf
   - Bar chart across environments: v_start_greedy_bars_across_envs.pdf
4. Summary table:
   - summary_metrics.csv
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"

ALL_ENVS = [
    "fourrooms-dense",
    "FourRooms-misc",
    "fourrooms-mines",
    "fourrooms-mines-dense",
    "eightrooms-dense",
    "eightrooms-misc",
    "whirlpool-misc",
    "SpaceInvadersExactValue",
    "mountaincar-dense",
]

ALGO_KEYS = ["mc", "td0", "td_lambda", "oct8"]

ALGO_COLORS = {
    "mc": "#1f77b4",          # Deep Blue
    "td0": "#9467bd",         # Purple
    "td_lambda": "#ff7f0e",   # Amber/Orange
    "oct8": "#2ca02c",        # Green
}

ALGO_LABELS = {
    "mc": "Sampled MC (λ=1.0)",
    "td0": "TD(0)",
    "td_lambda": "TD(λ)",
    "oct8": "E(0) (Oct 8)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Random Policy Benchmark Plots")
    parser.add_argument("--results_dir", type=str, default="results/random_policy_sweep", help="Results directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory")
    return parser.parse_args()


def load_env_results(results_dir: str, env_name: str):
    """Loads out.pkl for an environment containing dict of algorithm metrics."""
    candidate = os.path.join(results_dir, env_name, "out.pkl")
    if os.path.exists(candidate):
        try:
            with open(candidate, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {candidate}: {e}")
            return None
    return None


def extract_stats(arr):
    """Extracts mean and std over seeds."""
    if arr is None:
        return None, None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr, np.zeros_like(arr)
    elif arr.ndim >= 2:
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        if mean.ndim > 1:
            mean = mean.squeeze()
            std = std.squeeze()
        return mean, std
    return None, None


# ---------------------------------------------------------------------------
# 1. VALUE ERROR (VE)
# ---------------------------------------------------------------------------

def plot_ve_lines(env_data: dict, save_dir: str):
    """Line chart of Value Error MSE (nn_weighted_VE) during learning."""
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    n_cols = 3
    n_rows = int(np.ceil(n_envs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.4), dpi=200, sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for idx, env_name in enumerate(valid_envs):
        ax = axes[idx]
        data = env_data[env_name]

        for k in ALGO_KEYS:
            if k not in data:
                continue
            v_metrics = data[k]
            raw = v_metrics.get("nn_weighted_VE")
            if raw is None:
                continue

            mean, std = extract_stats(raw)
            if mean is not None:
                x = np.arange(len(mean))
                ax.plot(x, mean, color=ALGO_COLORS[k], lw=2.0, label=ALGO_LABELS[k])
                if std is not None and np.any(std > 0):
                    ax.fill_between(
                        x,
                        np.maximum(mean - std, 1e-12),
                        mean + std,
                        color=ALGO_COLORS[k],
                        alpha=0.18,
                    )

        ax.set_yscale("log")
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("VE MSE (Log Scale)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(n_envs, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=True)

    plt.suptitle("Value Error MSE During Learning Across Environments", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, "ve_lines_across_envs.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path}")


def plot_ve_bars(env_data: dict, save_dir: str):
    """Bar chart of average Value Error MSE during learning across environments."""
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    fig, ax = plt.subplots(figsize=(max(10, n_envs * 1.8), 6), dpi=200)

    x = np.arange(n_envs)
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for k, offset in zip(ALGO_KEYS, offsets):
        vals = []
        for e in valid_envs:
            data = env_data[e]
            if k in data and "nn_weighted_VE" in data[k]:
                m, _ = extract_stats(data[k]["nn_weighted_VE"])
                val = float(np.nanmean(m)) if m is not None else 1e-12
            else:
                val = 1e-12
            vals.append(max(val, 1e-12))

        ax.bar(
            x + offset,
            vals,
            width,
            label=ALGO_LABELS[k],
            color=ALGO_COLORS[k],
            alpha=0.88,
            edgecolor="black",
            lw=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(valid_envs, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Average VE MSE (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("Average Value Error MSE Across Environments", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(loc="upper right", ncol=4, fontsize=9.5, frameon=True)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "ve_bars_across_envs.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path}")


# ---------------------------------------------------------------------------
# 2. ADVANTAGE COSINE SIMILARITY
# ---------------------------------------------------------------------------

def plot_advantage_cossim_lines(env_data: dict, save_dir: str):
    """Line chart of Advantage Cosine Similarity (nn_advantage_cossim) during learning."""
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    n_cols = 3
    n_rows = int(np.ceil(n_envs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.4), dpi=200, sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for idx, env_name in enumerate(valid_envs):
        ax = axes[idx]
        data = env_data[env_name]

        for k in ALGO_KEYS:
            if k not in data:
                continue
            v_metrics = data[k]
            raw = v_metrics.get("nn_advantage_cossim")
            if raw is None:
                continue

            mean, std = extract_stats(raw)
            if mean is not None:
                x = np.arange(len(mean))
                ax.plot(x, mean, color=ALGO_COLORS[k], lw=2.0, label=ALGO_LABELS[k])
                if std is not None and np.any(std > 0):
                    ax.fill_between(
                        x,
                        mean - std,
                        mean + std,
                        color=ALGO_COLORS[k],
                        alpha=0.18,
                    )

        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("Advantage Cosine Sim", fontsize=9)
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(n_envs, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=True)

    plt.suptitle("Advantage Cosine Similarity During Learning Across Environments", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, "advantage_cossim_lines_across_envs.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path}")


def plot_advantage_cossim_bars(env_data: dict, save_dir: str):
    """Bar chart of final Advantage Cosine Similarity across environments."""
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    fig, ax = plt.subplots(figsize=(max(10, n_envs * 1.8), 6), dpi=200)

    x = np.arange(n_envs)
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for k, offset in zip(ALGO_KEYS, offsets):
        vals = []
        for e in valid_envs:
            data = env_data[e]
            if k in data and "nn_advantage_cossim" in data[k]:
                m, _ = extract_stats(data[k]["nn_advantage_cossim"])
                val = float(m[-1]) if m is not None else 0.0
            else:
                val = 0.0
            vals.append(val)

        ax.bar(
            x + offset,
            vals,
            width,
            label=ALGO_LABELS[k],
            color=ALGO_COLORS[k],
            alpha=0.88,
            edgecolor="black",
            lw=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(valid_envs, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax.set_ylabel("Final Advantage Cosine Sim", fontsize=11, fontweight="bold")
    ax.set_title("Final Advantage Cosine Similarity Across Environments", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(loc="upper left", ncol=4, fontsize=9.5, frameon=True)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "advantage_cossim_bars_across_envs.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path}")


# ---------------------------------------------------------------------------
# 3. V_START OF GREEDY POLICY
# ---------------------------------------------------------------------------

def plot_v_start_greedy_lines(env_data: dict, save_dir: str):
    """Line chart of V_start of the greedy policy (nn_greedy_performance) during learning."""
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    n_cols = 3
    n_rows = int(np.ceil(n_envs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.4), dpi=200, sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for idx, env_name in enumerate(valid_envs):
        ax = axes[idx]
        data = env_data[env_name]

        # Check for baseline V_start of random policy
        v_pi_start = None
        for k in ALGO_KEYS:
            if k in data and "V_start" in data[k]:
                v_start_raw = data[k]["V_start"]
                if v_start_raw is not None:
                    v_pi_start = float(np.asarray(v_start_raw).mean())
                    break

        for k in ALGO_KEYS:
            if k not in data:
                continue
            v_metrics = data[k]
            raw = v_metrics.get("nn_greedy_performance")
            if raw is None:
                continue

            mean, std = extract_stats(raw)
            if mean is not None:
                x = np.arange(len(mean))
                ax.plot(x, mean, color=ALGO_COLORS[k], lw=2.0, label=ALGO_LABELS[k])
                if std is not None and np.any(std > 0):
                    ax.fill_between(
                        x,
                        mean - std,
                        mean + std,
                        color=ALGO_COLORS[k],
                        alpha=0.18,
                    )

        if v_pi_start is not None:
            ax.axhline(v_pi_start, color="gray", linestyle=":", lw=1.5, label=r"Random Policy $V(s_0)$")

        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Update Step", fontsize=9)
        ax.set_ylabel("Greedy Policy V(s0)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(n_envs, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate legend entries
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), fontsize=9.5, frameon=True)

    plt.suptitle("Greedy Policy Start Value V(s0) During Learning Across Environments", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, "v_start_greedy_lines_across_envs.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path}")


def plot_v_start_greedy_bars(env_data: dict, save_dir: str):
    """Bar chart of final V_start of the greedy policy across environments."""
    valid_envs = [e for e in ALL_ENVS if e in env_data]
    if not valid_envs:
        return

    n_envs = len(valid_envs)
    fig, ax = plt.subplots(figsize=(max(10, n_envs * 1.8), 6), dpi=200)

    x = np.arange(n_envs)
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for k, offset in zip(ALGO_KEYS, offsets):
        vals = []
        for e in valid_envs:
            data = env_data[e]
            if k in data and "nn_greedy_performance" in data[k]:
                m, _ = extract_stats(data[k]["nn_greedy_performance"])
                val = float(m[-1]) if m is not None else 0.0
            else:
                val = 0.0
            vals.append(val)

        ax.bar(
            x + offset,
            vals,
            width,
            label=ALGO_LABELS[k],
            color=ALGO_COLORS[k],
            alpha=0.88,
            edgecolor="black",
            lw=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(valid_envs, fontsize=10, fontweight="bold", rotation=20, ha="right")
    ax.set_ylabel("Final Greedy Policy V(s0)", fontsize=11, fontweight="bold")
    ax.set_title("Final Greedy Policy Start Value Across Environments", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.legend(loc="upper left", ncol=4, fontsize=9.5, frameon=True)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "v_start_greedy_bars_across_envs.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path}")


# ---------------------------------------------------------------------------
# 4. SUMMARY CSV
# ---------------------------------------------------------------------------

def generate_summary_csv(env_data: dict, save_dir: str):
    """Generates consolidated metrics summary table in CSV format."""
    rows = []
    valid_envs = [e for e in ALL_ENVS if e in env_data]

    for env_name in valid_envs:
        row = {"Environment": env_name}
        data = env_data[env_name]

        for k in ALGO_KEYS:
            if k not in data:
                continue
            v_metrics = data[k]

            # 1. VE MSE
            if "nn_weighted_VE" in v_metrics:
                m, _ = extract_stats(v_metrics["nn_weighted_VE"])
                if m is not None:
                    row[f"{k}_VE_mean"] = float(np.nanmean(m))
                    row[f"{k}_VE_final"] = float(m[-1])

            # 2. Advantage Cosine Sim
            if "nn_advantage_cossim" in v_metrics:
                m, _ = extract_stats(v_metrics["nn_advantage_cossim"])
                if m is not None:
                    row[f"{k}_AdvCos_mean"] = float(np.nanmean(m))
                    row[f"{k}_AdvCos_final"] = float(m[-1])

            # 3. Greedy V_start
            if "nn_greedy_performance" in v_metrics:
                m, _ = extract_stats(v_metrics["nn_greedy_performance"])
                if m is not None:
                    row[f"{k}_GreedyVstart_mean"] = float(np.nanmean(m))
                    row[f"{k}_GreedyVstart_final"] = float(m[-1])

        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(save_dir, "summary_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved Summary Table: {csv_path}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    save_dir = os.path.abspath(args.save_dir if args.save_dir else results_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("Random Policy Benchmark: MC vs TD(0) vs TD(lambda) vs E(0)")
    print(f"Results Directory: {results_dir}")
    print(f"Save Directory:    {save_dir}")
    print("=" * 70)

    env_data = {}
    for env_name in ALL_ENVS:
        res = load_env_results(results_dir, env_name)
        if res is not None:
            print(f"Loaded: {env_name}")
            env_data[env_name] = res
        else:
            print(f"Missing: {env_name}")

    if not env_data:
        print("No environment data found in", results_dir)
        return

    # Visual 1: VE
    plot_ve_lines(env_data, save_dir)
    plot_ve_bars(env_data, save_dir)

    # Visual 2: Advantage Cosine Sim
    plot_advantage_cossim_lines(env_data, save_dir)
    plot_advantage_cossim_bars(env_data, save_dir)

    # Visual 3: Greedy V_start
    plot_v_start_greedy_lines(env_data, save_dir)
    plot_v_start_greedy_bars(env_data, save_dir)

    # Summary Table
    generate_summary_csv(env_data, save_dir)


if __name__ == "__main__":
    main()
