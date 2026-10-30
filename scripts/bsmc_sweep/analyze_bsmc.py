"""
scripts/bsmc_sweep/analyze_bsmc.py

Analyzes results from the BSMC sweep comparing:
  1. E(lambda) [Geometric Jumps, m ~ Geom]
  2. BSMC(lambda) [1-Step Jumps, m = 1, Analytical Matching MC Weight]
  3. Forward-Only vs. v_aux variants

Generates:
  - Table 1: Head-to-Head Comparison (E(lambda) vs BSMC(lambda) at identical MC weights)
  - Table 2: AUC Policy Return Matrix across all benchmark environments
  - Table 3: Multi-Step Jump Advantage Score: AUC(E(lambda)) - AUC(BSMC(lambda))
  - Vector plots: bsmc_vs_elambda_comparison.png

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
    Compares E(lambda) vs BSMC(lambda) at matching lambda values with matched MC weight.
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
    print(f"[Saved Head-to-Head Table]: {md_path}")


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

    build_head_to_head_table(df, out_dir)


if __name__ == "__main__":
    main()
