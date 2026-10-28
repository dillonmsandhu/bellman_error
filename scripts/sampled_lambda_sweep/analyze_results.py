"""
scripts/sampled_lambda_sweep/analyze_results.py

Comprehensive publication-focused data analysis for Sampled PPO Lambda Sweeps (TD(λ) vs E(λ) with v_aux).
Generates:
1. Table 1 (Matching-Lambda Head-to-Head): Decisive win/loss/tie records for λ ∈ {0.5, 0.9, 0.95}.
2. Table 2 (Pairwise Tournament Record): Pairwise decisive win-tie-loss matrix across all 6 conditions.
3. Table 3 (Tuned Critic LRs): Selected critic learning rates for each condition and environment.
4. Table 4 (AUC Return Matrix): Standardized AUC Policy Performance (Envs on rows, Conditions on columns).
5. Table 5 (Final Return Matrix): Final Standardized Policy Performance.
6. Table 6 (Value Error Matrix): Learned critic VE (nn_weighted_VE).
7. Table 7 (Advantage Cosine Similarity Matrix): Learned critic advantage alignment (nn_advantage_cossim).

Outputs all tables to CSV (.csv), Markdown (.md), LaTeX booktabs (.tex), and vector PDF (.pdf)
under <results_dir>/analysis_tables/.
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

# Publication font settings
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

CONDITION_ORDER = ["td_0.50", "td_0.90", "td_0.95", "e_0.50", "e_0.90", "e_0.95"]

CONDITION_DISPLAY_NAMES = {
    "td_0.50": "TD(0.50)",
    "td_0.90": "TD(0.90)",
    "td_0.95": "TD(0.95)",
    "e_0.50":  "E(0.50) [v_aux]",
    "e_0.90":  "E(0.90) [v_aux]",
    "e_0.95":  "E(0.95) [v_aux]",
}

MATCHING_LAMBDAS = [0.5, 0.9, 0.95]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Sampled PPO Lambda Sweep Results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/sampled_lambda_sweep",
        help="Root results directory containing environment subfolders",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for tables (default: <results_dir>/analysis_tables)",
    )
    return parser.parse_args()


def load_data(results_dir: str) -> pd.DataFrame:
    summary_path = os.path.join(results_dir, "summary_table_all.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
    else:
        dfs = []
        for env_name in ALL_ENVS:
            csv_path = os.path.join(results_dir, env_name, "summary.csv")
            if os.path.exists(csv_path):
                dfs.append(pd.read_csv(csv_path))
        if not dfs:
            raise FileNotFoundError(f"No summary.csv files found in {results_dir}")
        df = pd.concat(dfs, ignore_index=True)
    return df


def save_table_bundle(df: pd.DataFrame, table_id: str, title: str, out_dir: str, float_format: str = "%.4f"):
    """Saves a DataFrame as CSV, Markdown, LaTeX booktabs, and vector PDF."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. CSV
    csv_path = os.path.join(out_dir, f"{table_id}.csv")
    df.to_csv(csv_path, index=False)

    # 2. Markdown
    md_path = os.path.join(out_dir, f"{table_id}.md")
    with open(md_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    # 3. LaTeX Booktabs
    tex_path = os.path.join(out_dir, f"{table_id}.tex")
    latex_code = df.to_latex(index=False, float_format=float_format)
    with open(tex_path, "w") as f:
        f.write(f"% {title}\n")
        f.write(latex_code)

    # 4. Vector PDF
    pdf_path = os.path.join(out_dir, f"{table_id}.pdf")
    n_rows, n_cols = df.shape
    fig_w = max(8.5, min(24.0, n_cols * 1.8 + 2.0))
    fig_h = max(3.5, min(20.0, n_rows * 0.45 + 2.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    table_data = [df.columns.tolist()] + df.values.tolist()
    tbl = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f8f9fa")
        else:
            cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#dddddd")

    plt.tight_layout()
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  [OK] Exported {table_id} (.csv, .md, .tex, .pdf)")


def analyze_matching_lambda_head_to_head(df: pd.DataFrame, out_dir: str):
    """Table 1: Matching-lambda comparisons: E(λ) vs TD(λ) for λ in {0.5, 0.9, 0.95}."""
    rows = []

    for lam in MATCHING_LAMBDAS:
        lam_str = f"{lam:.2f}"
        td_key = f"td_{lam_str}"
        e_key = f"e_{lam_str}"

        td_wins = 0
        e_wins = 0
        ties = 0

        td_auc_list = []
        e_auc_list = []
        deltas = []

        for env_name in ALL_ENVS:
            sub = df[df["env_name"] == env_name]
            td_row = sub[sub["condition"] == td_key]
            e_row = sub[sub["condition"] == e_key]

            if td_row.empty or e_row.empty:
                continue

            td_val = float(td_row["std_perf_auc"].values[0])
            td_sem = float(td_row.get("std_perf_final_sem", pd.Series([0.0])).values[0])

            e_val = float(e_row["std_perf_auc"].values[0])
            e_sem = float(e_row.get("std_perf_final_sem", pd.Series([0.0])).values[0])

            td_auc_list.append(td_val)
            e_auc_list.append(e_val)
            deltas.append(e_val - td_val)

            diff = e_val - td_val
            err_margin = td_sem + e_sem
            if abs(diff) <= err_margin:
                ties += 1
            elif diff > 0:
                e_wins += 1
            else:
                td_wins += 1

        n_envs = len(td_auc_list)
        if n_envs > 0:
            mean_td = np.mean(td_auc_list)
            mean_e = np.mean(e_auc_list)
            mean_delta = np.mean(deltas)
            win_rate = (e_wins / n_envs) * 100.0

            rows.append({
                "Lambda (λ)": f"{lam:.2f}",
                "E(λ) [v_aux] Mean AUC": f"{mean_e:+.4f}",
                "TD(λ) Mean AUC": f"{mean_td:+.4f}",
                "Mean Advantage (E - TD)": f"{mean_delta:+.4f}",
                "E Wins": e_wins,
                "Ties": ties,
                "TD Wins": td_wins,
                "E Win Rate (%)": f"{win_rate:.1f}%",
            })

    table_df = pd.DataFrame(rows)
    save_table_bundle(
        table_df,
        table_id="table1_matching_lambda_head_to_head",
        title="Table 1: Matching-Lambda Head-to-Head Comparison [E(λ) vs TD(λ)]",
        out_dir=out_dir,
    )


def compute_pairwise_record(df: pd.DataFrame, out_dir: str):
    """Table 2: Pairwise tournament win-tie-loss record across all 6 conditions."""
    conds = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    n = len(conds)
    matrix = np.empty((n, n), dtype=object)

    for i, c1 in enumerate(conds):
        for j, c2 in enumerate(conds):
            if i == j:
                matrix[i, j] = "-"
                continue

            wins, ties, losses = 0, 0, 0
            for env_name in ALL_ENVS:
                sub = df[df["env_name"] == env_name]
                r1 = sub[sub["condition"] == c1]
                r2 = sub[sub["condition"] == c2]

                if r1.empty or r2.empty:
                    continue

                v1 = float(r1["std_perf_auc"].values[0])
                s1 = float(r1.get("std_perf_final_sem", pd.Series([0.0])).values[0])

                v2 = float(r2["std_perf_auc"].values[0])
                s2 = float(r2.get("std_perf_final_sem", pd.Series([0.0])).values[0])

                diff = v1 - v2
                err = s1 + s2
                if abs(diff) <= err:
                    ties += 1
                elif diff > 0:
                    wins += 1
                else:
                    losses += 1

            matrix[i, j] = f"{wins}-{ties}-{losses}"

    table_df = pd.DataFrame(matrix, index=[CONDITION_DISPLAY_NAMES.get(c, c) for c in conds], columns=[CONDITION_DISPLAY_NAMES.get(c, c) for c in conds])
    table_df.reset_index(inplace=True)
    table_df.rename(columns={"index": "Condition (Row vs Col)"}, inplace=True)

    save_table_bundle(
        table_df,
        table_id="table2_pairwise_tournament_record",
        title="Table 2: Pairwise Decisive Record Matrix (Wins - Ties - Losses)",
        out_dir=out_dir,
    )


def build_matrix_table(df: pd.DataFrame, metric_col: str, title: str, table_id: str, out_dir: str, fmt: str = "{:+.4f}"):
    """Builds an (Environments x Conditions) performance matrix."""
    conds = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    envs = [e for e in ALL_ENVS if e in df["env_name"].unique()]

    rows = []
    col_vals = {c: [] for c in conds}

    for env_name in envs:
        row = {"Environment": ENV_DISPLAY_NAMES.get(env_name, env_name)}
        sub = df[df["env_name"] == env_name]

        for c in conds:
            c_row = sub[sub["condition"] == c]
            if not c_row.empty:
                val = float(c_row[metric_col].values[0])
                col_vals[c].append(val)
                row[CONDITION_DISPLAY_NAMES.get(c, c)] = fmt.format(val)
            else:
                row[CONDITION_DISPLAY_NAMES.get(c, c)] = "N/A"
        rows.append(row)

    # Average row
    avg_row = {"Environment": "Average (All Envs)"}
    for c in conds:
        if col_vals[c]:
            avg_row[CONDITION_DISPLAY_NAMES.get(c, c)] = fmt.format(np.mean(col_vals[c]))
        else:
            avg_row[CONDITION_DISPLAY_NAMES.get(c, c)] = "N/A"
    rows.append(avg_row)

    table_df = pd.DataFrame(rows)
    save_table_bundle(table_df, table_id=table_id, title=title, out_dir=out_dir)


def build_critic_lr_table(df: pd.DataFrame, out_dir: str):
    """Table 3: Tuned Critic Learning Rates selected across conditions."""
    conds = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    envs = [e for e in ALL_ENVS if e in df["env_name"].unique()]

    rows = []
    for env_name in envs:
        row = {"Environment": ENV_DISPLAY_NAMES.get(env_name, env_name)}
        sub = df[df["env_name"] == env_name]
        for c in conds:
            c_row = sub[sub["condition"] == c]
            if not c_row.empty and "critic_lr" in c_row.columns:
                lr = c_row["critic_lr"].values[0]
                row[CONDITION_DISPLAY_NAMES.get(c, c)] = str(lr)
            else:
                row[CONDITION_DISPLAY_NAMES.get(c, c)] = "N/A"
        rows.append(row)

    table_df = pd.DataFrame(rows)
    save_table_bundle(
        table_df,
        table_id="table3_tuned_critic_lrs",
        title="Table 3: Tuned Critic Learning Rates Across Environments",
        out_dir=out_dir,
    )


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(results_dir, "analysis_tables"))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("Sampled PPO Lambda Sweep: Publication Data Analysis")
    print(f"Results Directory: {results_dir}")
    print(f"Tables Directory:  {out_dir}")
    print("=" * 80)

    df = load_data(results_dir)
    print(f"Loaded {len(df)} condition rows across {df['env_name'].nunique()} environments.")

    # 1. Matching-Lambda Head-to-Head Table
    analyze_matching_lambda_head_to_head(df, out_dir)

    # 2. Pairwise Tournament Decisive Record
    compute_pairwise_record(df, out_dir)

    # 3. Selected Critic Learning Rates
    build_critic_lr_table(df, out_dir)

    # 4. AUC Return Matrix
    build_matrix_table(
        df,
        metric_col="std_perf_auc",
        title="Table 4: Standardized Area Under Curve (AUC) Policy Return Matrix",
        table_id="table4_std_perf_auc_matrix",
        out_dir=out_dir,
    )

    # 5. Final Return Matrix
    build_matrix_table(
        df,
        metric_col="std_perf_final",
        title="Table 5: Final Standardized Policy Return Matrix",
        table_id="table5_std_perf_final_matrix",
        out_dir=out_dir,
    )

    # 6. Critic Value Error Matrix
    if "ve_final" in df.columns:
        build_matrix_table(
            df,
            metric_col="ve_final",
            title="Table 6: Learned Critic Final Value Error Matrix (nn_weighted_VE)",
            table_id="table6_critic_value_error_matrix",
            out_dir=out_dir,
            fmt="{:.4f}",
        )

    # 7. Advantage Cosine Similarity Matrix
    if "cossim_final" in df.columns:
        build_matrix_table(
            df,
            metric_col="cossim_final",
            title="Table 7: Learned Critic Advantage Cosine Similarity Matrix",
            table_id="table7_advantage_cossim_matrix",
            out_dir=out_dir,
            fmt="{:+.4f}",
        )

    print("\nAll analysis tables generated successfully!")


if __name__ == "__main__":
    main()
