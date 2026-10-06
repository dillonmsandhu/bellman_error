#!/usr/bin/env python3
"""
sweep_expected_layernorm.py
Hyperparameter sweep over Learning Rate and Layer Normalization for Expected PPOs:
1. exact_E   (Symmetrized Bellman error gradient / Dirichlet form)
2. exact_td  (Expected TD(0) semi-gradient)
3. exact_mc  (Expected Monte Carlo mean squared error gradient)

Generates 3 individual plots per environment (one per gradient):
- With vs Without Layer Norm, evaluated at the same chosen LR
  (selected as the LR that worked best for both, or barring that the best LR for no layer norm).
- Metric plotted: Start-state value V_start = V^pi(s_0) under the learning policy.
Also generates a consolidated 1x3 panel comparison figure.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Publication-grade vector PDF embedding
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"

import core.config as default_cfg
from ppo.exact_E import make_train as make_train_E
from ppo.exact_td import make_train as make_train_TD
from ppo.exact_mc import make_train as make_train_MC

ALL_ENVS = [
    "fourrooms-dense",
    "FourRooms-misc",
    "eightrooms-dense",
    "eightrooms-misc",
    "whirlpool-misc",
    "SpaceInvadersExactValue",
    "mountaincar-dense",
]

ALGO_MAP = {
    "exact_E": (make_train_E, "Exact Bellman Error ($g_E^*$)"),
    "exact_td": (make_train_TD, "Exact TD(0) ($g_{\\mathrm{TD}}^*$)"),
    "exact_mc": (make_train_MC, "Exact Monte Carlo ($g_{\\mathrm{MC}}^*$)"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep LR and LayerNorm for Expected PPOs (exact_E, exact_td, exact_mc)"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Name of the environment (e.g. FourRooms-misc, fourrooms-dense)",
    )
    parser.add_argument(
        "--env_idx",
        type=int,
        default=None,
        help="Index into ALL_ENVS (0 to 6)",
    )
    parser.add_argument(
        "--algos",
        type=str,
        nargs="+",
        default=["exact_E", "exact_td", "exact_mc"],
        choices=["exact_E", "exact_td", "exact_mc"],
        help="Algorithms to evaluate",
    )
    parser.add_argument(
        "--lr_grid",
        type=float,
        nargs="+",
        default=[0.005, 0.001, 0.0003, 0.0001],
        help="Grid of learning rates to sweep",
    )
    parser.add_argument(
        "--num_updates",
        type=int,
        default=300,
        help="Number of policy updates / timesteps",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=4,
        help="Number of random seeds per configuration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=30,
        help="Number of final steps to average when determining best LR",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots and NPZ artifacts",
    )
    parser.add_argument(
        "--only_plot",
        action="store_true",
        help="Skip sweep and regenerate plots from existing .npz files",
    )
    return parser.parse_args()


def run_configuration(
    algo_name: str,
    env_name: str,
    lr: float,
    layer_norm: bool,
    num_updates: int,
    num_seeds: int,
    base_seed: int,
) -> np.ndarray:
    """Runs a single (algo, lr, layer_norm) configuration across seeds via JAX vmap."""
    make_train_fn, _ = ALGO_MAP[algo_name]

    cfg = default_cfg.config.copy()
    cfg.update({
        "ENV_NAME": env_name,
        "TOTAL_TIMESTEPS": num_updates,
        "NUM_UPDATES": num_updates,
        "LAYER_NORM": bool(layer_norm),
        "LR": float(lr),
        "ACTOR_LR": float(lr),
        "CALC_TRUE_VALUES": True,
        "LIGHT_METRICS": True,
        "LOG_FEATURE_METRICS": False,
    })

    train_fn = make_train_fn(cfg)
    vmapped_train = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))

    rng = jax.random.PRNGKey(base_seed)
    rngs = jax.random.split(rng, num_seeds)

    t0 = time.time()
    out = vmapped_train(rngs, None)
    # Block until computation is finished
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        out,
    )
    elapsed = time.time() - t0

    # Extract V_start across seeds: shape (num_seeds, num_updates)
    v_start = np.array(out["metrics"]["V_start"])
    print(
        f"    [{algo_name}] LN={str(layer_norm):<5} | LR={lr:<7} | "
        f"Final V_start: {v_start[:, -1].mean():.4f} +/- {v_start[:, -1].std():.4f} "
        f"({elapsed:.1f}s)"
    )
    return v_start


def select_best_lr(
    scores_no_ln: Dict[float, float],
    scores_with_ln: Dict[float, float],
) -> Tuple[float, float, float, str]:
    """
    Selects the LR for comparison according to the rule:
    1. The LR that worked best for both (if they agree).
    2. Barring that, the LR that worked best for no layer norm (layer_norm=False).
    """
    best_lr_no_ln = max(scores_no_ln, key=scores_no_ln.get)
    best_lr_with_ln = max(scores_with_ln, key=scores_with_ln.get)

    if np.isclose(best_lr_no_ln, best_lr_with_ln):
        chosen_lr = best_lr_no_ln
        reason = f"Concordant: best for both conditions ({chosen_lr})"
    else:
        chosen_lr = best_lr_no_ln
        reason = (
            f"Discordant: LN=False preferred {best_lr_no_ln}, LN=True preferred {best_lr_with_ln}. "
            f"Defaulted to LN=False best ({chosen_lr})"
        )

    return chosen_lr, best_lr_no_ln, best_lr_with_ln, reason


def plot_single_gradient_comparison(
    algo_name: str,
    algo_title: str,
    env_name: str,
    chosen_lr: float,
    traj_no_ln: np.ndarray,
    traj_with_ln: np.ndarray,
    reason: str,
    out_dir: str,
):
    """Plots and exports a single gradient's comparison (with vs without LayerNorm)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)

    num_seeds, num_updates = traj_no_ln.shape
    x = np.arange(1, num_updates + 1)

    # Compute mean and SEM
    mean_no_ln = np.mean(traj_no_ln, axis=0)
    sem_no_ln = np.std(traj_no_ln, axis=0) / np.sqrt(num_seeds)

    mean_with_ln = np.mean(traj_with_ln, axis=0)
    sem_with_ln = np.std(traj_with_ln, axis=0) / np.sqrt(num_seeds)

    final_no_ln = mean_no_ln[-1]
    final_with_ln = mean_with_ln[-1]

    # Style definitions
    c_no_ln = "#1f77b4"     # Steel Blue
    c_with_ln = "#d62728"   # Crimson / Red

    ax.plot(
        x,
        mean_no_ln,
        color=c_no_ln,
        linewidth=2.2,
        label=f"No LayerNorm (final={final_no_ln:.3f})",
        zorder=3,
    )
    ax.fill_between(
        x,
        mean_no_ln - sem_no_ln,
        mean_no_ln + sem_no_ln,
        color=c_no_ln,
        alpha=0.20,
        zorder=2,
    )

    ax.plot(
        x,
        mean_with_ln,
        color=c_with_ln,
        linewidth=2.2,
        label=f"With LayerNorm (final={final_with_ln:.3f})",
        zorder=4,
    )
    ax.fill_between(
        x,
        mean_with_ln - sem_with_ln,
        mean_with_ln + sem_with_ln,
        color=c_with_ln,
        alpha=0.20,
        zorder=2,
    )

    ax.set_xlabel("Policy Update Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Start State Value $V^\\pi(s_0)$", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{env_name} — {algo_title}\n(Learning Rate = {chosen_lr})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="lower right")

    # Add small explanatory footnote
    fig.text(
        0.5,
        -0.02,
        f"Selection Note: {reason}",
        ha="center",
        fontsize=8,
        color="#555555",
        style="italic",
    )

    fig.tight_layout()

    pdf_path = os.path.join(out_dir, f"{env_name}_{algo_name}_layernorm.pdf")
    png_path = os.path.join(out_dir, f"{env_name}_{algo_name}_layernorm.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot Saved] {pdf_path}")


def plot_consolidated_panel(
    env_name: str,
    algos: List[str],
    chosen_lrs: Dict[str, float],
    trajectories_no_ln: Dict[str, np.ndarray],
    trajectories_with_ln: Dict[str, np.ndarray],
    out_dir: str,
):
    """Plots a 1x3 comparative panel across all three expected gradients."""
    fig, axes = plt.subplots(1, len(algos), figsize=(5.5 * len(algos), 4.5), dpi=300, sharey=True)
    if len(algos) == 1:
        axes = [axes]

    c_no_ln = "#1f77b4"
    c_with_ln = "#d62728"

    for idx, algo in enumerate(algos):
        ax = axes[idx]
        _, algo_title = ALGO_MAP[algo]
        lr = chosen_lrs[algo]

        t_no = trajectories_no_ln[algo]
        t_with = trajectories_with_ln[algo]
        num_seeds, num_updates = t_no.shape
        x = np.arange(1, num_updates + 1)

        m_no = np.mean(t_no, axis=0)
        s_no = np.std(t_no, axis=0) / np.sqrt(num_seeds)

        m_with = np.mean(t_with, axis=0)
        s_with = np.std(t_with, axis=0) / np.sqrt(num_seeds)

        ax.plot(
            x,
            m_no,
            color=c_no_ln,
            linewidth=2.0,
            label=f"No LN (final={m_no[-1]:.3f})",
        )
        ax.fill_between(x, m_no - s_no, m_no + s_no, color=c_no_ln, alpha=0.20)

        ax.plot(
            x,
            m_with,
            color=c_with_ln,
            linewidth=2.0,
            label=f"With LN (final={m_with[-1]:.3f})",
        )
        ax.fill_between(x, m_with - s_with, m_with + s_with, color=c_with_ln, alpha=0.20)

        ax.set_title(f"{algo_title}\n(LR = {lr})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Policy Update Step", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Start State Value $V^\\pi(s_0)$", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9, loc="lower right")

    fig.suptitle(f"Expected PPO LayerNorm Comparison: {env_name}", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()

    pdf_path = os.path.join(out_dir, f"{env_name}_expected_layernorm_comparison.pdf")
    png_path = os.path.join(out_dir, f"{env_name}_expected_layernorm_comparison.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Consolidated Panel Saved] {pdf_path}")


def main():
    args = parse_args()

    # Resolve environment
    if args.env_name is not None:
        env_name = args.env_name
    elif args.env_idx is not None:
        if args.env_idx < 0 or args.env_idx >= len(ALL_ENVS):
            raise ValueError(f"env_idx {args.env_idx} out of range [0, {len(ALL_ENVS) - 1}]")
        env_name = ALL_ENVS[args.env_idx]
    else:
        env_name = "FourRooms-misc"

    out_dir = args.out_dir or os.path.join("results", "sweeps", "expected_layernorm", env_name)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{env_name}_layernorm_sweep.npz")
    json_path = os.path.join(out_dir, f"{env_name}_summary.json")

    print("=" * 70)
    print("EXPECTED PPO LAYER NORM & LR SWEEP")
    print(f"Environment:       {env_name}")
    print(f"Algorithms:        {args.algos}")
    print(f"Learning Rate Grid:{args.lr_grid}")
    print(f"Layer Norms:       [False, True]")
    print(f"Policy Updates:    {args.num_updates}")
    print(f"Seeds:             {args.num_seeds} (base: {args.seed})")
    print(f"Output Directory:  {out_dir}")
    print("=" * 70)

    # Store all runs: raw_results[algo][layer_norm][lr] -> array (seeds, updates)
    raw_results: Dict[str, Dict[bool, Dict[float, np.ndarray]]] = {
        algo: {False: {}, True: {}} for algo in args.algos
    }

    if args.only_plot and os.path.exists(npz_path):
        print(f"Loading existing data from {npz_path}...")
        loaded = np.load(npz_path)
        for key in loaded.files:
            # key format: {algo}__ln_{ln}__lr_{lr}
            parts = key.split("__")
            if len(parts) == 3:
                algo = parts[0]
                ln = parts[1].split("_")[1].lower() == "true"
                lr = float(parts[2].split("_")[1])
                if algo in raw_results:
                    raw_results[algo][ln][lr] = loaded[key]
    else:
        # Run sweep
        for algo in args.algos:
            print(f"\n---> Sweeping Algorithm: {algo}")
            for ln in [False, True]:
                for lr in args.lr_grid:
                    v_start = run_configuration(
                        algo_name=algo,
                        env_name=env_name,
                        lr=lr,
                        layer_norm=ln,
                        num_updates=args.num_updates,
                        num_seeds=args.num_seeds,
                        base_seed=args.seed,
                    )
                    raw_results[algo][ln][lr] = v_start

        # Save to NPZ
        npz_dict = {}
        for algo in args.algos:
            for ln in [False, True]:
                for lr, data in raw_results[algo][ln].items():
                    npz_dict[f"{algo}__ln_{ln}__lr_{lr}"] = data
        np.savez_compressed(npz_path, **npz_dict)
        print(f"\n[Artifact Saved] Raw sweep arrays saved to {npz_path}")

    # Process results: Select best LR for each algorithm and generate plots
    print("\n" + "=" * 70)
    print("EVALUATING AND SELECTING BEST LEARNING RATES")
    print("=" * 70)

    summary_records = {}
    chosen_lrs = {}
    best_traj_no_ln = {}
    best_traj_with_ln = {}

    w = max(1, args.window_size)

    for algo in args.algos:
        scores_no_ln = {}
        scores_with_ln = {}

        for lr in args.lr_grid:
            if lr in raw_results[algo][False]:
                # Mean performance across seeds over the final window
                scores_no_ln[lr] = float(np.mean(raw_results[algo][False][lr][:, -w:]))
            if lr in raw_results[algo][True]:
                scores_with_ln[lr] = float(np.mean(raw_results[algo][True][lr][:, -w:]))

        chosen_lr, best_lr_no, best_lr_with, reason = select_best_lr(scores_no_ln, scores_with_ln)
        chosen_lrs[algo] = chosen_lr

        best_traj_no_ln[algo] = raw_results[algo][False][chosen_lr]
        best_traj_with_ln[algo] = raw_results[algo][True][chosen_lr]

        _, algo_title = ALGO_MAP[algo]
        print(f"\n[{algo}]")
        print(f"  Scores without LN: {scores_no_ln}")
        print(f"  Scores with LN:    {scores_with_ln}")
        print(f"  Selected LR:       {chosen_lr} ({reason})")

        summary_records[algo] = {
            "algo_title": algo_title,
            "chosen_lr": chosen_lr,
            "best_lr_no_ln": best_lr_no,
            "best_lr_with_ln": best_lr_with,
            "selection_reason": reason,
            "scores_no_ln": scores_no_ln,
            "scores_with_ln": scores_with_ln,
            "final_V_start_no_ln_mean": float(np.mean(best_traj_no_ln[algo][:, -1])),
            "final_V_start_no_ln_sem": float(np.std(best_traj_no_ln[algo][:, -1]) / np.sqrt(args.num_seeds)),
            "final_V_start_with_ln_mean": float(np.mean(best_traj_with_ln[algo][:, -1])),
            "final_V_start_with_ln_sem": float(np.std(best_traj_with_ln[algo][:, -1]) / np.sqrt(args.num_seeds)),
        }

        # Plot single gradient figure (1 plot per gradient)
        plot_single_gradient_comparison(
            algo_name=algo,
            algo_title=algo_title,
            env_name=env_name,
            chosen_lr=chosen_lr,
            traj_no_ln=best_traj_no_ln[algo],
            traj_with_ln=best_traj_with_ln[algo],
            reason=reason,
            out_dir=out_dir,
        )

    # Save summary JSON
    with open(json_path, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\n[Summary JSON Saved] {json_path}")

    # Plot consolidated 1x3 multi-panel figure
    plot_consolidated_panel(
        env_name=env_name,
        algos=args.algos,
        chosen_lrs=chosen_lrs,
        trajectories_no_ln=best_traj_no_ln,
        trajectories_with_ln=best_traj_with_ln,
        out_dir=out_dir,
    )

    print("\n" + "=" * 70)
    print("ALL PLOTS AND SUMMARIES GENERATED SUCCESSFULLY")
    print(f"Location: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
