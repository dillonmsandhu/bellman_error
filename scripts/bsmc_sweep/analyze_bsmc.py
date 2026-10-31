"""
scripts/bsmc_sweep/analyze_bsmc.py

Analyzes results from the BSMC sweep comparing:
  1. E(lambda) [Geometric Jumps, m ~ Geom]
  2. BSMC(lambda) [1-Step Jumps, m = 1, Analytical Matching MC Weight]
  3. Forward-Only vs. v_aux variants

Generates:
  - Table 1: Head-to-Head Comparison (E(lambda) vs BSMC(lambda) at identical MC weights)
  - Table 2: Per-Environment Policy Return Matrix across all 9 benchmark environments
  - Table 3: Per-Environment Jump Advantage Score: AUC(E(lambda)) - AUC(BSMC(lambda))
  - Publication vector figures:
      * bsmc_vs_elambda_overview.pdf (and .png)
      * bsmc_jump_advantage_by_env.pdf (and .png)

Outputs to <results_dir>/analysis_tables/.
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matplotlib publication formatting
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
    "eightrooms-misc",
    "eightrooms-dense",
    "whirlpool-misc",
    "SpaceInvadersExactValue",
    "mountaincar-dense",
]

ENV_DISPLAY_NAMES = {
    "mountaincar": "Mountain Car (Sparse)",
    "FourRooms-misc": "FourRooms (Sparse)",
    "fourrooms-mines": "FourRooms + Mines",
    "fourrooms-mines-dense": "FourRooms + Mines (Dense)",
    "eightrooms-misc": "EightRooms (Sparse)",
    "eightrooms-dense": "EightRooms (Dense)",
    "whirlpool-misc": "Whirlpool (Irreversible)",
    "SpaceInvadersExactValue": "Space Invaders",
    "mountaincar-dense": "Mountain Car (Dense)",
}

MATCHING_LAMBDAS = [0.4, 0.8, 0.95, 0.99]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze BSMC vs E(lambda) Disentanglement Results")
    parser.add_argument("--results_dir", type=str, default="results/bsmc_sweep", help="Results directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for tables and plots")
    return parser.parse_args()


def load_all_summaries(results_dir: str) -> pd.DataFrame:
    dfs = []
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory {results_dir} does not exist.")
        return pd.DataFrame()

    for item in os.listdir(results_dir):
        env_dir = os.path.join(results_dir, item)
        if os.path.isdir(env_dir):
            csv_path = os.path.join(env_dir, "summary.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if "Environment" not in df.columns:
                        df["Environment"] = item
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def build_head_to_head_table(df: pd.DataFrame, out_dir: str):
    """
    Table 1: Head-to-Head Comparison (E(lambda) vs BSMC(lambda) at identical MC weights).
    """
    rows = []
    for lam in MATCHING_LAMBDAS:
        # v_aux branch
        e_sub = df[(df["Algorithm"] == "e_lambda") & (np.isclose(df["Lambda"], lam))]
        bsmc_sub = df[(df["Algorithm"] == "bsmc_aux") & (np.isclose(df["Lambda"], lam))]

        common_envs = sorted(list(set(e_sub["Environment"]).intersection(set(bsmc_sub["Environment"]))))
        if common_envs:
            e_vals = e_sub.set_index("Environment").loc[common_envs, "Std_Perf_AUC_Mean"].values
            bsmc_vals = bsmc_sub.set_index("Environment").loc[common_envs, "Std_Perf_AUC_Mean"].values

            diff = e_vals - bsmc_vals
            e_wins = int(np.sum(diff > 0.01))
            bsmc_wins = int(np.sum(diff < -0.01))
            ties = len(common_envs) - e_wins - bsmc_wins
            mean_diff = float(np.mean(diff))

            mc_pct = float(e_sub["MC_Weight_Pct"].iloc[0]) if "MC_Weight_Pct" in e_sub.columns else (1.0 - 0.99) / (1.0 - 0.99 * lam) * 100.0

            rows.append({
                "Variant": "v_aux Debiased",
                "Lambda": lam,
                "MC Weight (%)": f"{mc_pct:.2f}%",
                "E(λ) Mean AUC": f"{np.mean(e_vals):.4f}",
                "BSMC(λ) Mean AUC": f"{np.mean(bsmc_vals):.4f}",
                "Jump Advantage [E - BSMC]": f"{mean_diff:+.4f}",
                "Record (E / Tie / BSMC)": f"{e_wins} / {ties} / {bsmc_wins}",
            })

        # Forward-Only branch
        e_fwd_sub = df[(df["Algorithm"] == "e_lambda_fwd") & (np.isclose(df["Lambda"], lam))]
        bsmc_fwd_sub = df[(df["Algorithm"] == "bsmc_fwd") & (np.isclose(df["Lambda"], lam))]

        common_envs_fwd = sorted(list(set(e_fwd_sub["Environment"]).intersection(set(bsmc_fwd_sub["Environment"]))))
        if common_envs_fwd:
            e_fwd_vals = e_fwd_sub.set_index("Environment").loc[common_envs_fwd, "Std_Perf_AUC_Mean"].values
            bsmc_fwd_vals = bsmc_fwd_sub.set_index("Environment").loc[common_envs_fwd, "Std_Perf_AUC_Mean"].values

            diff_fwd = e_fwd_vals - bsmc_fwd_vals
            e_fwd_wins = int(np.sum(diff_fwd > 0.01))
            bsmc_fwd_wins = int(np.sum(diff_fwd < -0.01))
            ties_fwd = len(common_envs_fwd) - e_fwd_wins - bsmc_fwd_wins
            mean_diff_fwd = float(np.mean(diff_fwd))

            mc_pct = float(e_fwd_sub["MC_Weight_Pct"].iloc[0]) if "MC_Weight_Pct" in e_fwd_sub.columns else (1.0 - 0.99) / (1.0 - 0.99 * lam) * 100.0

            rows.append({
                "Variant": "Forward-Only",
                "Lambda": lam,
                "MC Weight (%)": f"{mc_pct:.2f}%",
                "E(λ) Mean AUC": f"{np.mean(e_fwd_vals):.4f}",
                "BSMC(λ) Mean AUC": f"{np.mean(bsmc_fwd_vals):.4f}",
                "Jump Advantage [E - BSMC]": f"{mean_diff_fwd:+.4f}",
                "Record (E / Tie / BSMC)": f"{e_fwd_wins} / {ties_fwd} / {bsmc_fwd_wins}",
            })

    if not rows:
        return
    res_df = pd.DataFrame(rows)
    md_path = os.path.join(out_dir, "table1_head_to_head_e_vs_bsmc.md")
    csv_path = os.path.join(out_dir, "table1_head_to_head_e_vs_bsmc.csv")
    res_df.to_markdown(md_path, index=False)
    res_df.to_csv(csv_path, index=False)
    print(f"[Saved Table 1 (Head-to-Head)]: {md_path}")


def build_per_env_matrix(df: pd.DataFrame, out_dir: str):
    """
    Table 2: Per-Environment Policy Return Matrix across all 9 benchmark environments.
    """
    envs = [e for e in ALL_ENVS if e in df["Environment"].unique()]
    if not envs:
        envs = sorted(df["Environment"].unique().tolist())

    rows = []
    for env in envs:
        row = {"Environment": ENV_DISPLAY_NAMES.get(env, env)}
        for lam in MATCHING_LAMBDAS:
            for algo, prefix in [("e_lambda", "E"), ("bsmc_aux", "BSMC"), ("td_lambda", "TD")]:
                sub = df[(df["Environment"] == env) & (df["Algorithm"] == algo) & (np.isclose(df["Lambda"], lam))]
                if not sub.empty:
                    val = float(sub["Std_Perf_AUC_Mean"].iloc[0])
                    row[f"{prefix}({lam})"] = f"{val:.4f}"
                else:
                    row[f"{prefix}({lam})"] = "-"
        rows.append(row)

    if not rows:
        return
    matrix_df = pd.DataFrame(rows)
    md_path = os.path.join(out_dir, "table2_per_env_auc_matrix.md")
    csv_path = os.path.join(out_dir, "table2_per_env_auc_matrix.csv")
    matrix_df.to_markdown(md_path, index=False)
    matrix_df.to_csv(csv_path, index=False)
    print(f"[Saved Table 2 (Per-Env Matrix)]: {md_path}")


def build_advantage_delta_table(df: pd.DataFrame, out_dir: str):
    """
    Table 3: Per-Environment Multi-Step Jump Advantage Score: AUC(E) - AUC(BSMC).
    """
    envs = [e for e in ALL_ENVS if e in df["Environment"].unique()]
    if not envs:
        envs = sorted(df["Environment"].unique().tolist())

    rows = []
    for env in envs:
        row = {"Environment": ENV_DISPLAY_NAMES.get(env, env)}
        for lam in MATCHING_LAMBDAS:
            e_sub = df[(df["Environment"] == env) & (df["Algorithm"] == "e_lambda") & (np.isclose(df["Lambda"], lam))]
            b_sub = df[(df["Environment"] == env) & (df["Algorithm"] == "bsmc_aux") & (np.isclose(df["Lambda"], lam))]
            if not e_sub.empty and not b_sub.empty:
                delta = float(e_sub["Std_Perf_AUC_Mean"].iloc[0]) - float(b_sub["Std_Perf_AUC_Mean"].iloc[0])
                row[f"Δ_v_aux({lam})"] = f"{delta:+.4f}"
            else:
                row[f"Δ_v_aux({lam})"] = "-"

            # Forward-only
            ef_sub = df[(df["Environment"] == env) & (df["Algorithm"] == "e_lambda_fwd") & (np.isclose(df["Lambda"], lam))]
            bf_sub = df[(df["Environment"] == env) & (df["Algorithm"] == "bsmc_fwd") & (np.isclose(df["Lambda"], lam))]
            if not ef_sub.empty and not bf_sub.empty:
                delta_f = float(ef_sub["Std_Perf_AUC_Mean"].iloc[0]) - float(bf_sub["Std_Perf_AUC_Mean"].iloc[0])
                row[f"Δ_fwd({lam})"] = f"{delta_f:+.4f}"
            else:
                row[f"Δ_fwd({lam})"] = "-"

        rows.append(row)

    if not rows:
        return
    delta_df = pd.DataFrame(rows)
    md_path = os.path.join(out_dir, "table3_jump_advantage_deltas.md")
    csv_path = os.path.join(out_dir, "table3_jump_advantage_deltas.csv")
    delta_df.to_markdown(md_path, index=False)
    delta_df.to_csv(csv_path, index=False)
    print(f"[Saved Table 3 (Advantage Deltas)]: {md_path}")


def plot_comparison_figures(df: pd.DataFrame, out_dir: str):
    """
    Generates publication vector PDF and PNG charts.
    """
    # Figure 1: Mean AUC vs Lambda across formulations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Panel A: v_aux Debiased: E(lambda) vs BSMC(lambda) vs TD(lambda)
    ax1 = axes[0]
    lambdas = sorted(df["Lambda"].unique())
    
    e_means, bsmc_means, td_means = [], [], []
    for lam in lambdas:
        e_s = df[(df["Algorithm"] == "e_lambda") & (np.isclose(df["Lambda"], lam))]["Std_Perf_AUC_Mean"]
        b_s = df[(df["Algorithm"] == "bsmc_aux") & (np.isclose(df["Lambda"], lam))]["Std_Perf_AUC_Mean"]
        td_s = df[(df["Algorithm"] == "td_lambda") & (np.isclose(df["Lambda"], lam))]["Std_Perf_AUC_Mean"]
        e_means.append(float(e_s.mean()) if not e_s.empty else np.nan)
        bsmc_means.append(float(b_s.mean()) if not b_s.empty else np.nan)
        td_means.append(float(td_s.mean()) if not td_s.empty else np.nan)

    ax1.plot(lambdas, e_means, marker="o", color="#2ca02c", linewidth=2.2, label=r"Geometric $E(\lambda)$ [$v_{\rm aux}$]")
    ax1.plot(lambdas, bsmc_means, marker="s", color="#9467bd", linewidth=2.0, linestyle="--", label=r"1-Step BSMC($\lambda$) [$v_{\rm aux}$]")
    if any(not np.isnan(v) for v in td_means):
        ax1.plot(lambdas, td_means, marker="^", color="#ff7f0e", linewidth=1.8, linestyle=":", label=r"TD($\lambda$)")

    ax1.set_title(r"A. $v_{\rm aux}$ Debiased: Multi-Step vs 1-Step", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel(r"Trace Parameter $\lambda$", fontsize=11)
    ax1.set_ylabel("Mean Standardized AUC Return", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    # Panel B: Forward-Only: E(lambda) vs BSMC(lambda)
    ax2 = axes[1]
    ef_means, bf_means = [], []
    for lam in lambdas:
        ef_s = df[(df["Algorithm"] == "e_lambda_fwd") & (np.isclose(df["Lambda"], lam))]["Std_Perf_AUC_Mean"]
        bf_s = df[(df["Algorithm"] == "bsmc_fwd") & (np.isclose(df["Lambda"], lam))]["Std_Perf_AUC_Mean"]
        ef_means.append(float(ef_s.mean()) if not ef_s.empty else np.nan)
        bf_means.append(float(bf_s.mean()) if not bf_s.empty else np.nan)

    ax2.plot(lambdas, ef_means, marker="o", color="#1f77b4", linewidth=2.2, label=r"Geometric $E(\lambda)$ [Forward-Only]")
    ax2.plot(lambdas, bf_means, marker="s", color="#e377c2", linewidth=2.0, linestyle="--", label=r"1-Step BSMC($\lambda$) [Forward-Only]")

    ax2.set_title("B. Forward-Only: Multi-Step vs 1-Step", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel(r"Trace Parameter $\lambda$", fontsize=11)
    ax2.set_ylabel("Mean Standardized AUC Return", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "bsmc_vs_elambda_overview.pdf")
    png_path = os.path.join(out_dir, "bsmc_vs_elambda_overview.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Saved Overview Plots]: {pdf_path}")

    # Figure 2: Jump Advantage by Environment at lambda = 0.80 (where advantage peaks)
    envs = [e for e in ALL_ENVS if e in df["Environment"].unique()]
    if not envs:
        envs = sorted(df["Environment"].unique().tolist())

    diff_08 = []
    env_labels = []
    for env in envs:
        e_s = df[(df["Environment"] == env) & (df["Algorithm"] == "e_lambda") & (np.isclose(df["Lambda"], 0.8))]
        b_s = df[(df["Environment"] == env) & (df["Algorithm"] == "bsmc_aux") & (np.isclose(df["Lambda"], 0.8))]
        if not e_s.empty and not b_s.empty:
            diff_08.append(float(e_s["Std_Perf_AUC_Mean"].iloc[0]) - float(b_s["Std_Perf_AUC_Mean"].iloc[0]))
            env_labels.append(ENV_DISPLAY_NAMES.get(env, env))

    if diff_08:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 4.5), dpi=300)
        y_pos = np.arange(len(env_labels))
        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in diff_08]

        ax_bar.barh(y_pos, diff_08, color=colors, alpha=0.85, edgecolor="#333333", height=0.6)
        ax_bar.axvline(0.0, color="#333333", linestyle="-", linewidth=0.8)
        ax_bar.axvline(0.01, color="#2ca02c", linestyle=":", alpha=0.6, label="Win Threshold (+0.01)")
        ax_bar.axvline(-0.01, color="#d62728", linestyle=":", alpha=0.6, label="Loss Threshold (-0.01)")

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(env_labels, fontsize=10)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel(r"Jump Advantage: ${\rm AUC}(E(0.80)) - {\rm AUC}({\rm BSMC}(0.80))$", fontsize=11)
        ax_bar.set_title(r"Multi-Step Jump Advantage by Environment ($\lambda = 0.80, v_{\rm aux}$ Debiased)", fontsize=12, fontweight="bold", pad=10)
        ax_bar.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax_bar.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

        plt.tight_layout()
        pdf_bar = os.path.join(out_dir, "bsmc_jump_advantage_by_env.pdf")
        png_bar = os.path.join(out_dir, "bsmc_jump_advantage_by_env.png")
        fig_bar.savefig(pdf_bar, bbox_inches="tight")
        fig_bar.savefig(png_bar, bbox_inches="tight", dpi=300)
        plt.close(fig_bar)
        print(f"[Saved Advantage by Environment Bar Plot]: {pdf_bar}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = args.out_dir or os.path.join(results_dir, "analysis_tables")
    os.makedirs(out_dir, exist_ok=True)

    df = load_all_summaries(results_dir)
    if df.empty:
        print(f"No summary CSV files found in {results_dir}. Run the sweep first.")
        return

    print("=" * 80)
    print("BSMC vs E(lambda) Disentanglement Analysis")
    print(f"Results Directory: {results_dir}")
    print(f"Output Directory:  {out_dir}")
    print(f"Loaded {len(df)} total evaluated condition rows across {df['Environment'].nunique()} environments.")
    print("=" * 80)

    # 1. Table 1: Head-to-Head summary
    build_head_to_head_table(df, out_dir)

    # 2. Table 2: Per-environment matrix
    build_per_env_matrix(df, out_dir)

    # 3. Table 3: Per-environment advantage deltas
    build_advantage_delta_table(df, out_dir)

    # 4. Publication Vector Plots
    plot_comparison_figures(df, out_dir)

    print("\n[Done] All BSMC analysis tables and vector figures generated successfully.")


if __name__ == "__main__":
    main()
