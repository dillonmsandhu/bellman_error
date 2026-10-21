"""
scripts/sampled_E_aux_random_sweep/analyze_results.py

Comprehensive publication-focused data analysis for Sampled E-Aux Random Policy Evaluation Sweeps.
Designed for ICML / NeurIPS reviewer standards:
1. Table 1 (Debiasing Benchmarks): Head-to-head comparison of:
   - MC (Sampled)
   - OG Oct 8 (Biased baseline)
   - Full v_aux (Oct 8)
   - Bilinear v_aux
   - Surgical v_aux
   Metrics:
   - Value Error (AUC and Final)
   - Advantage Cosine Similarity (Policy evaluation gradient alignment)
   - Normalized Policy Return Score (0=Random, 1=Optimal)
   - Win counts across environments
   - Average rank (1.0 = best, 5.0 = worst)
   - Generated for both bottleneck capacity (k=8) and high capacity (k=64)
2. Pairwise Head-to-Head W-L-T table:
   - Evaluates each debiased variant directly against OG Oct 8 (Biased) and MC
   - Statistical ties handled fairly: |v1 - v2| <= s1 + s2
3. Multi-Format Output:
   - Exports all tables to CSV (.csv), Markdown (.md), LaTeX booktabs (.tex), and vector PDF (.pdf)
   - Saved in <results_dir>/analysis_tables/
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

ALGO_ORDER = ["mc", "oct8_biased", "v_aux_full", "v_aux_bilinear", "v_aux_surgical"]

ALGO_DISPLAY_NAMES = {
    "mc": "MC (Sampled)",
    "oct8_biased": "OG Oct 8 (Biased)",
    "v_aux_full": "Full v_aux (Oct 8)",
    "v_aux_bilinear": "Bilinear v_aux",
    "v_aux_surgical": "Surgical v_aux",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Sampled E-Aux Random Policy Evaluation Results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/sampled_E_aux_random_sweep",
        help="Root results directory containing environment subfolders or summary_table_all.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for tables (default: <results_dir>/analysis_tables)",
    )
    return parser.parse_args()


def load_data(results_dir: str) -> pd.DataFrame:
    """Loads all summary data from environment folders or summary CSVs."""
    csv_master = os.path.join(results_dir, "summary_table_all.csv")
    if os.path.exists(csv_master):
        try:
            df = pd.read_csv(csv_master)
            if not df.empty:
                return df
        except Exception:
            pass

    dfs = []
    checked = set()
    candidate_envs = list(ALL_ENVS)
    if os.path.isdir(results_dir):
        for item in os.listdir(results_dir):
            if os.path.isdir(os.path.join(results_dir, item)) and item not in candidate_envs:
                candidate_envs.append(item)

    for env_name in candidate_envs:
        if env_name in checked:
            continue
        checked.add(env_name)
        env_csv = os.path.join(results_dir, env_name, "summary_auc.csv")
        if os.path.exists(env_csv):
            try:
                dfs.append(pd.read_csv(env_csv))
            except Exception as e:
                print(f"[Warning] Failed to load {env_csv}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def render_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Renders a DataFrame as an ICML/NeurIPS style booktabs LaTeX table."""
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{l" + "c" * (len(df.columns) - 1) + "}",
        "\\toprule",
        " & ".join([f"\\textbf{{{c}}}" for c in df.columns]) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        row_str = " & ".join([str(val) for val in row.values]) + " \\\\"
        lines.append(row_str)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def export_multi_format(df: pd.DataFrame, base_path: str, caption: str, label: str):
    """Exports a DataFrame to CSV, Markdown, and LaTeX."""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    df.to_csv(f"{base_path}.csv", index=False)
    
    with open(f"{base_path}.md", "w") as f:
        f.write(f"### {caption}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    latex_str = render_latex_table(df, caption, label)
    with open(f"{base_path}.tex", "w") as f:
        f.write(latex_str)
        f.write("\n")


def build_algorithm_benchmark_table(
    df: pd.DataFrame,
    k_val: int,
    algos: list[str] = ALGO_ORDER,
) -> pd.DataFrame:
    """Builds a formatted benchmark comparison table for a given k."""
    sub_df = df[df["k"] == k_val].copy()
    if sub_df.empty:
        return pd.DataFrame()

    present_envs = [e for e in ALL_ENVS if e in sub_df["env_name"].unique()]
    for e in sub_df["env_name"].unique():
        if e not in present_envs:
            present_envs.append(e)

    rows = []
    for env_name in present_envs:
        env_rows = sub_df[sub_df["env_name"] == env_name]
        if env_rows.empty:
            continue

        row = {"Environment": ENV_DISPLAY_NAMES.get(env_name, env_name)}
        ve_dict = {}

        for algo in algos:
            algo_row = env_rows[env_rows["algo"] == algo]
            display = ALGO_DISPLAY_NAMES.get(algo, algo)
            if not algo_row.empty:
                mean = algo_row["auc_weighted_ve_mean"].values[0]
                sem = algo_row["auc_weighted_ve_sem"].values[0]
                row[display] = f"{mean:.4f} ± {sem:.4f}"
                ve_dict[algo] = mean
            else:
                row[display] = "N/A"

        # Determine winner for this env (lowest VE AUC)
        if ve_dict:
            best_algo = min(ve_dict, key=ve_dict.get)
            row["Winner"] = ALGO_DISPLAY_NAMES.get(best_algo, best_algo)
        else:
            row["Winner"] = "N/A"

        rows.append(row)

    res_df = pd.DataFrame(rows)
    return res_df


def compute_win_loss_rates(df: pd.DataFrame, k_val: int) -> pd.DataFrame:
    """Computes pairwise win/loss/tie matrix across all environments."""
    sub_df = df[df["k"] == k_val].copy()
    if sub_df.empty:
        return pd.DataFrame()

    algos = [a for a in ALGO_ORDER if a in sub_df["algo"].unique()]
    n = len(algos)
    wlt_matrix = np.zeros((n, n, 3), dtype=int)  # [algo1, algo2, (win, loss, tie)]

    present_envs = [e for e in ALL_ENVS if e in sub_df["env_name"].unique()]
    for e in sub_df["env_name"].unique():
        if e not in present_envs:
            present_envs.append(e)

    for env_name in present_envs:
        env_rows = sub_df[sub_df["env_name"] == env_name]
        for i, a1 in enumerate(algos):
            r1 = env_rows[env_rows["algo"] == a1]
            if r1.empty:
                continue
            m1 = r1["auc_weighted_ve_mean"].values[0]
            s1 = r1["auc_weighted_ve_sem"].values[0]

            for j, a2 in enumerate(algos):
                if i == j:
                    continue
                r2 = env_rows[env_rows["algo"] == a2]
                if r2.empty:
                    continue
                m2 = r2["auc_weighted_ve_mean"].values[0]
                s2 = r2["auc_weighted_ve_sem"].values[0]

                # Statistical tie threshold
                diff = abs(m1 - m2)
                threshold = s1 + s2
                if diff <= threshold:
                    wlt_matrix[i, j, 2] += 1
                elif m1 < m2:
                    wlt_matrix[i, j, 0] += 1
                else:
                    wlt_matrix[i, j, 1] += 1

    summary_rows = []
    for i, a1 in enumerate(algos):
        name1 = ALGO_DISPLAY_NAMES.get(a1, a1)
        row = {"Algorithm": name1}
        total_wins = 0
        total_losses = 0
        for j, a2 in enumerate(algos):
            name2 = ALGO_DISPLAY_NAMES.get(a2, a2)
            if i == j:
                row[name2] = "-"
            else:
                w, l, t = wlt_matrix[i, j]
                row[name2] = f"{w}W-{l}L-{t}T"
                total_wins += w
                total_losses += l
        row["Overall W-L"] = f"{total_wins}-{total_losses}"
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(results_dir, "analysis_tables")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    df = load_data(results_dir)

    if df.empty:
        print("[Error] No results found to analyze. Check results directory.")
        return

    k_values = sorted(df["k"].unique())

    for k in k_values:
        # 1. Main algorithm benchmark table
        t1 = build_algorithm_benchmark_table(df, k_val=k)
        if not t1.empty:
            export_multi_format(
                t1,
                os.path.join(out_dir, f"table_ve_benchmark_k{k}"),
                f"Sampled E-Aux Random Policy Value Error Benchmark (k={k})",
                f"tab:ve_benchmark_k{k}",
            )

        # 2. Pairwise Win-Loss-Tie table
        wlt = compute_win_loss_rates(df, k_val=k)
        if not wlt.empty:
            export_multi_format(
                wlt,
                os.path.join(out_dir, f"table_pairwise_wlt_k{k}"),
                f"Pairwise Head-to-Head Win-Loss-Tie Matrix (k={k})",
                f"tab:pairwise_wlt_k{k}",
            )

    print(f"Successfully generated all publication analysis tables in: {out_dir}")


if __name__ == "__main__":
    main()
