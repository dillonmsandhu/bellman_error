"""
scripts/exact_ppo_sweep/analyze_results.py

Comprehensive Data Analysis and Publication Table Generator for Exact PPO Sweep.
Generates:
1. Table 1: Final Standardized Policy Performance (Environments on Rows, Algorithms on Columns)
   - Highest performing algorithm in each environment highlighted in bold.
   - Fair statistical ties considered: if |v_best - v| <= SEM_best + SEM, co-winners are bolded.
   - Summary rows: Mean Performance and Win Count across environments.
2. Table 2: Area Under Curve (AUC) Standardized Policy Performance
3. Table 3: Pairwise Head-to-Head Matrix [Wins - Losses - Ties] across environments.

All tables exported in 4 publication formats:
- CSV (.csv)
- Markdown (.md)
- LaTeX (.tex) with booktabs
- Rendered publication vector PDF (.pdf)
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

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

ENV_DISPLAY_NAMES = {
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

ALGO_DISPLAY_NAMES = {
    "mc": "MC (Exact)",
    "td0": "TD(0)",
    "e0": "E(0)",
    "td_lambda": "TD(λ=0.8)",
    "e_lambda": "E(λ=0.8)",
}

ALGO_ORDER = ["mc", "td0", "e0", "td_lambda", "e_lambda"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Exact PPO Sweep Results and Generate Publication Tables")
    parser.add_argument("--results_dir", type=str, default="results/exact_ppo_sweep", help="Root directory containing env results")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save tables (default: results_dir/analysis_tables)")
    return parser.parse_args()


def load_dataset(results_dir: str):
    """Loads checkpoints and summaries across all environments."""
    env_data = {}
    summary_dfs = []

    for env in ALL_ENVS:
        env_dir = os.path.join(results_dir, env)
        pkl_path = os.path.join(env_dir, "out.pkl")
        csv_path = os.path.join(env_dir, "summary.csv")

        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    env_data[env] = pickle.load(f)
            except Exception as e:
                print(f"[Warning] Failed loading {pkl_path}: {e}")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                summary_dfs.append(df)
            except Exception as e:
                print(f"[Warning] Failed loading {csv_path}: {e}")

    combined_df = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()
    return env_data, combined_df


def generate_standardized_performance_table(
    summary_df: pd.DataFrame,
    metric_prefix: str = "std_perf_final",
    table_title: str = "Standardized Policy Performance",
):
    """
    Generates a table with environments on rows, algorithms on columns.
    Winner in each environment row in bold, accounting for statistical ties (within 1 SEM).
    """
    mean_col = f"{metric_prefix}_mean"
    sem_col = f"{metric_prefix}_sem"

    envs_present = [e for e in ALL_ENVS if e in summary_df["env"].unique()]
    algos_present = [a for a in ALGO_ORDER if a in summary_df["algo"].unique()]

    if not envs_present or not algos_present:
        return pd.DataFrame(), {}

    data_matrix = []
    bold_cells = {}  # (row_idx, col_idx) -> bool
    win_counts = {a: 0.0 for a in algos_present}

    for row_idx, env in enumerate(envs_present):
        row = {"Environment": ENV_DISPLAY_NAMES.get(env, env)}
        env_sub = summary_df[summary_df["env"] == env]

        row_vals = {}
        row_sems = {}
        for a in algos_present:
            sub = env_sub[env_sub["algo"] == a]
            if not sub.empty:
                v = float(sub[mean_col].values[0])
                s = float(sub[sem_col].values[0]) if sem_col in sub else 0.0
                row_vals[a] = v
                row_sems[a] = s
            else:
                row_vals[a] = np.nan
                row_sems[a] = 0.0

        # Find best algorithm (highest performance)
        valid_algos = [a for a in algos_present if not np.isnan(row_vals[a])]
        if valid_algos:
            best_algo = max(valid_algos, key=lambda a: row_vals[a])
            best_val = row_vals[best_algo]
            best_sem = row_sems[best_algo]

            # Identify winners and statistical ties: |best_val - v| <= best_sem + s
            winners = []
            for a in valid_algos:
                val = row_vals[a]
                sem = row_sems[a]
                if (best_val - val) <= (best_sem + sem + 1e-6):
                    winners.append(a)

            # Credit win splits
            for w in winners:
                win_counts[w] += 1.0 / len(winners)

            # Mark bold cells
            for col_idx, a in enumerate(algos_present):
                col_name = ALGO_DISPLAY_NAMES.get(a, a)
                v = row_vals[a]
                s = row_sems[a]
                if np.isnan(v):
                    row[col_name] = "-"
                else:
                    row[col_name] = f"{v:.3f} ± {s:.3f}"
                    if a in winners:
                        bold_cells[(row_idx, col_idx + 1)] = True
        else:
            for col_idx, a in enumerate(algos_present):
                col_name = ALGO_DISPLAY_NAMES.get(a, a)
                row[col_name] = "-"

        data_matrix.append(row)

    # Summary Row: Overall Mean Performance
    mean_row = {"Environment": "Mean Performance"}
    win_row = {"Environment": f"Envs Won (out of {len(envs_present)})"}

    for col_idx, a in enumerate(algos_present):
        col_name = ALGO_DISPLAY_NAMES.get(a, a)
        sub = summary_df[summary_df["algo"] == a]
        if not sub.empty:
            vals = sub[mean_col].values
            avg = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            mean_row[col_name] = f"{avg:.3f} ± {sem:.3f}"
        else:
            mean_row[col_name] = "-"

        win_row[col_name] = f"{win_counts[a]:.1f}/{len(envs_present)}"

    data_matrix.append(mean_row)
    data_matrix.append(win_row)

    df_out = pd.DataFrame(data_matrix)
    return df_out, bold_cells


def generate_pairwise_head_to_head_table(
    summary_df: pd.DataFrame,
    metric_col: str = "std_perf_final_mean",
    sem_col: str = "std_perf_final_sem",
):
    """Generates pairwise Head-to-Head win-loss-tie matrix between all algorithms."""
    algos_present = [a for a in ALGO_ORDER if a in summary_df["algo"].unique()]
    envs_present = [e for e in ALL_ENVS if e in summary_df["env"].unique()]

    if not algos_present or not envs_present:
        return pd.DataFrame(), {}

    data_matrix = []
    bold_cells = {}

    for row_idx, a_row in enumerate(algos_present):
        row_display = ALGO_DISPLAY_NAMES.get(a_row, a_row)
        row = {"Algorithm": row_display}

        for col_idx, a_col in enumerate(algos_present):
            col_display = ALGO_DISPLAY_NAMES.get(a_col, a_col)
            if a_row == a_col:
                row[col_display] = "-"
                continue

            wins = 0
            losses = 0
            ties = 0

            for env in envs_present:
                sub_row = summary_df[(summary_df["env"] == env) & (summary_df["algo"] == a_row)]
                sub_col = summary_df[(summary_df["env"] == env) & (summary_df["algo"] == a_col)]

                if sub_row.empty or sub_col.empty:
                    continue

                v_row = float(sub_row[metric_col].values[0])
                s_row = float(sub_row[sem_col].values[0]) if sem_col in sub_row else 0.0

                v_col = float(sub_col[metric_col].values[0])
                s_col = float(sub_col[sem_col].values[0]) if sem_col in sub_col else 0.0

                diff = v_row - v_col
                combined_sem = s_row + s_col

                if abs(diff) <= combined_sem + 1e-6:
                    ties += 1
                elif diff > 0:
                    wins += 1
                else:
                    losses += 1

            row[col_display] = f"{wins}W-{losses}L-{ties}T"
            if wins > losses:
                bold_cells[(row_idx, col_idx + 1)] = True

        data_matrix.append(row)

    df_out = pd.DataFrame(data_matrix)
    return df_out, bold_cells


def export_table_formats(
    df: pd.DataFrame,
    bold_cells: dict,
    title: str,
    base_filename: str,
    out_dir: str,
):
    """Exports a dataframe to CSV, Markdown, LaTeX, and rendered vector PDF."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{base_filename}.csv")
    md_path = os.path.join(out_dir, f"{base_filename}.md")
    tex_path = os.path.join(out_dir, f"{base_filename}.tex")
    pdf_path = os.path.join(out_dir, f"{base_filename}.pdf")

    # 1. CSV
    df.to_csv(csv_path, index=False)

    # 2. Markdown
    df_md = df.copy()
    for (r, c), is_bold in bold_cells.items():
        if is_bold and r < len(df_md):
            col_name = df_md.columns[c]
            val = str(df_md.iloc[r, c])
            if val != "-":
                df_md.iloc[r, c] = f"**{val}**"
    with open(md_path, "w") as f:
        f.write(f"### {title}\n\n")
        f.write(df_md.to_markdown(index=False))
        f.write("\n")

    # 3. LaTeX
    df_tex = df.copy()
    for (r, c), is_bold in bold_cells.items():
        if is_bold and r < len(df_tex):
            val = str(df_tex.iloc[r, c])
            if val != "-":
                df_tex.iloc[r, c] = f"\\textbf{{{val}}}"
    latex_str = df_tex.to_latex(index=False, escape=False)
    with open(tex_path, "w") as f:
        f.write(f"% {title}\n")
        f.write(latex_str)

    # 4. Rendered Publication Vector PDF
    render_table_to_pdf(df, bold_cells, title, pdf_path)
    print(f"[Exported] {base_filename} -> CSV, MD, TEX, PDF")


def render_table_to_pdf(df: pd.DataFrame, bold_cells: dict, title: str, out_pdf_path: str):
    """Renders a formatted table directly to a vector PDF using matplotlib."""
    n_rows, n_cols = df.shape
    fig_width = max(10, n_cols * 2.2)
    fig_height = max(4.5, n_rows * 0.45 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table_data = []
    headers = list(df.columns)

    for r_idx in range(n_rows):
        row_vals = []
        for c_idx in range(n_cols):
            val = str(df.iloc[r_idx, c_idx])
            row_vals.append(val)
        table_data.append(row_vals)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.45)

    # Style header and cells
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2b5c8f")
            cell.set_text_props(color="white", weight="bold")
        elif r == n_rows or r == n_rows - 1:
            cell.set_facecolor("#f0f4f8")
            cell.set_text_props(weight="bold")
        else:
            if (r - 1, c) in bold_cells:
                cell.set_text_props(weight="bold", color="#0b6623")
                cell.set_facecolor("#eef8ee")
            elif r % 2 == 1:
                cell.set_facecolor("#f9f9f9")
            else:
                cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.6)

    plt.title(title, fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(out_pdf_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(results_dir, "analysis_tables")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Exact PPO Sweep Table Analysis Generator")
    print(f"Results Directory: {results_dir}")
    print(f"Tables Directory:  {out_dir}")
    print("=" * 70)

    env_data, summary_df = load_dataset(results_dir)
    print(f"Loaded {len(env_data)} environments.")

    if summary_df.empty:
        print("[Notice] No completed summary.csv files found yet. Analysis script is ready to process completed runs.")
        return

    # Table 1: Final Standardized Policy Performance
    df_t1, bold_t1 = generate_standardized_performance_table(
        summary_df=summary_df,
        metric_prefix="std_perf_final",
        table_title="Final Standardized Policy Performance",
    )
    export_table_formats(
        df=df_t1,
        bold_cells=bold_t1,
        title="Table 1: Final Standardized Policy Performance (Random=0, Optimal=1)",
        base_filename="table1_final_standardized_performance",
        out_dir=out_dir,
    )

    # Table 2: AUC Standardized Policy Performance
    df_t2, bold_t2 = generate_standardized_performance_table(
        summary_df=summary_df,
        metric_prefix="std_perf_auc",
        table_title="AUC Standardized Policy Performance",
    )
    export_table_formats(
        df=df_t2,
        bold_cells=bold_t2,
        title="Table 2: Area Under Curve (AUC) Standardized Policy Performance",
        base_filename="table2_auc_standardized_performance",
        out_dir=out_dir,
    )

    # Table 3: Pairwise Head-to-Head Win/Loss/Tie Records
    df_t3, bold_t3 = generate_pairwise_head_to_head_table(
        summary_df=summary_df,
        metric_col="std_perf_final_mean",
        sem_col="std_perf_final_sem",
    )
    export_table_formats(
        df=df_t3,
        bold_cells=bold_t3,
        title="Table 3: Pairwise Head-to-Head Records [Wins - Losses - Ties]",
        base_filename="table3_pairwise_head_to_head",
        out_dir=out_dir,
    )

    print("\n[Complete] All Exact PPO publication tables generated successfully.")


if __name__ == "__main__":
    main()
