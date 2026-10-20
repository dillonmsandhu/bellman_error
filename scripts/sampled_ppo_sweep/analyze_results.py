"""
scripts/sampled_ppo_sweep/analyze_results.py

Comprehensive publication-focused data analysis for Sampled PPO Sweeps.
Designed for ICML / NeurIPS reviewer standards:
1. Table 1 (Core Baselines): Head-to-head comparison of MC, TD(0), and E(0) [Oct 8, λ=1]
2. Table 2 (E(0) Variants Triad): Focused comparison of E(0) [Sept 14], E(0) [Oct 5], and E(0) [Oct 8]
3. Table 3 (Full 5 Spectrum): Full spectrum across all 5 algorithms
4. Table 4 (Pairwise Tournament Record): Head-to-head decisive records with statistical ties (|v1 - v2| <= s1 + s2)
5. Table 5 (Tuned Critic LRs): Selected critic learning rates across algorithms and environments
6. Table 6 (AUC Return Matrix): Area Under Curve (AUC) Standardized Policy Performance (Envs on rows, Algos on cols)
7. Table 7 (Final Return Matrix): Final Standardized Policy Performance (Envs on rows, Algos on cols)
8. Table 8 (E(0) Formulations & Return Lambdas): Multi-variant breakdown comparing all 6 E(0) conditions (3 variants x 2 lambdas)

Outputs all tables to CSV (.csv), Markdown (.md), LaTeX booktabs (.tex), and vector PDF (.pdf)
under <results_dir>/analysis_tables/.
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

ALGO_ORDER = ["mc", "td0", "e0_sep14", "e0_oct5", "e0_oct8"]

ALGO_DISPLAY_NAMES = {
    "mc": "MC (Sampled)",
    "td0": "TD(0) [Sampled]",
    "e0_sep14": "E(0) [Sept 14 / Uncorrected]",
    "e0_oct5": "E(0) [Oct 5 / Corrected]",
    "e0_oct8": "E(0) [Oct 8 / Start-Anchored]",
    "e0_sep14_lam1": "E(0) [Sept 14, λ=1]",
    "e0_oct5_lam1": "E(0) [Oct 5, λ=1]",
    "e0_oct8_lam1": "E(0) [Oct 8, λ=1]",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Sampled PPO Evaluation Results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/sampled_ppo_sweep",
        help="Root results directory containing environment subfolders or summary.csv files",
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
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError(f"No summary CSVs found in {results_dir}")

    if "env_name" in df.columns and "env" not in df.columns:
        df["env"] = df["env_name"]

    return df


def save_table_outputs(
    df: pd.DataFrame,
    out_dir: str,
    base_name: str,
    title: str,
    bold_cells: dict | None = None,
):
    """Exports a dataframe to CSV, Markdown, LaTeX booktabs, and vector PDF."""
    os.makedirs(out_dir, exist_ok=True)
    bold_dict = bold_cells if bold_cells is not None else {}

    # 1. Raw CSV
    csv_file = os.path.join(out_dir, f"{base_name}.csv")
    df.to_csv(csv_file, index=False)

    # 2. Markdown with bolding
    md_file = os.path.join(out_dir, f"{base_name}.md")
    df_md = df.copy()
    for (r, c), is_bold in bold_dict.items():
        if is_bold and r < len(df_md) and c < len(df_md.columns):
            val = str(df_md.iloc[r, c])
            if val != "-":
                df_md.iloc[r, c] = f"**{val}**"
    with open(md_file, "w") as f:
        f.write(f"### {title}\n\n")
        f.write(df_md.to_markdown(index=False))
        f.write("\n")

    # 3. LaTeX with booktabs and bolding
    tex_file = os.path.join(out_dir, f"{base_name}.tex")
    df_tex = df.copy()
    for (r, c), is_bold in bold_dict.items():
        if is_bold and r < len(df_tex) and c < len(df_tex.columns):
            val = str(df_tex.iloc[r, c])
            if val != "-":
                df_tex.iloc[r, c] = f"\\textbf{{{val}}}"
    latex_str = df_tex.to_latex(index=False, escape=False)
    with open(tex_file, "w") as f:
        f.write(f"% {title}\n")
        f.write(latex_str)

    # 4. Render vector PDF table
    pdf_file = os.path.join(out_dir, f"{base_name}.pdf")
    render_table_to_pdf(df, pdf_file, title, bold_cells=bold_dict)
    print(f"[Table Saved] {base_name} -> {md_file}, {tex_file}, {pdf_file}")


def render_table_to_pdf(
    df: pd.DataFrame,
    pdf_path: str,
    title: str,
    bold_cells: dict | None = None,
):
    """Renders a pandas DataFrame as a publication-grade vector PDF table."""
    bold_dict = bold_cells if bold_cells is not None else {}
    n_rows, n_cols = df.shape
    fig_width = max(10, n_cols * 2.2)
    fig_height = max(4.0, 0.45 * (n_rows + 4))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    clean_cols = [str(c).replace("_", " ").title() for c in df.columns]
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([str(val) if pd.notna(val) else "-" for val in row.values])

    table = ax.table(
        cellText=cell_text,
        colLabels=clean_cols,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.05, 1.45)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(weight="bold", color="#ffffff")
            cell.set_facecolor("#2b5c8f")
        elif r == n_rows or r == n_rows - 1:
            # Summary rows (Mean Performance, Envs Won)
            cell.set_facecolor("#f0f4f8")
            cell.set_text_props(weight="bold")
        else:
            if (r - 1, c) in bold_dict and bold_dict[(r - 1, c)]:
                cell.set_text_props(weight="bold", color="#0b6623")
                cell.set_facecolor("#eef8ee")
            elif r % 2 == 1:
                cell.set_facecolor("#f9fbfd")
            else:
                cell.set_facecolor("#ffffff")

    plt.title(title, fontsize=12, fontweight="bold", pad=14, loc="center")
    plt.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def generate_env_by_algo_table(
    summary_df: pd.DataFrame,
    algo_keys: list[str],
    metric_prefix: str = "std_perf_auc",
) -> tuple[pd.DataFrame, dict]:
    """
    Generates a table with environments on rows and algorithms on columns.
    Winner in each environment row in bold, accounting for statistical ties (within 1 combined SEM).
    Includes summary rows for Mean Performance and Envs Won.
    """
    mean_col = f"{metric_prefix}_mean"
    sem_col = f"{metric_prefix}_sem"

    envs_present = [e for e in ALL_ENVS if e in summary_df["env"].unique()]
    algos_present = [a for a in algo_keys if a in summary_df["algo"].unique()]

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
                s = float(sub[sem_col].values[0]) if sem_col in sub.columns else 0.0
                row_vals[a] = v
                row_sems[a] = s
            else:
                row_vals[a] = np.nan
                row_sems[a] = 0.0

        valid_algos = [a for a in algos_present if not np.isnan(row_vals[a])]
        if valid_algos:
            best_algo = max(valid_algos, key=lambda a: row_vals[a])
            best_val = row_vals[best_algo]
            best_sem = row_sems[best_algo]

            # Winners and statistical ties: (best_val - val) <= best_sem + sem
            winners = []
            for a in valid_algos:
                val = row_vals[a]
                sem = row_sems[a]
                if (best_val - val) <= (best_sem + sem + 1e-6):
                    winners.append(a)

            # Split wins evenly
            for w in winners:
                win_counts[w] += 1.0 / len(winners)

            # Format cells
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

    # Summary Row 1: Overall Mean Performance
    mean_row = {"Environment": "Mean Performance"}
    for a in algos_present:
        col_name = ALGO_DISPLAY_NAMES.get(a, a)
        sub = summary_df[summary_df["algo"] == a]
        if not sub.empty:
            vals = sub[mean_col].values
            avg = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            mean_row[col_name] = f"{avg:.3f} ± {sem:.3f}"
        else:
            mean_row[col_name] = "-"
    data_matrix.append(mean_row)

    # Summary Row 2: Envs Won (accounting for shared ties)
    win_row = {"Environment": f"Envs Won (out of {len(envs_present)})"}
    for a in algos_present:
        col_name = ALGO_DISPLAY_NAMES.get(a, a)
        win_row[col_name] = f"{win_counts[a]:.1f}/{len(envs_present)}"
    data_matrix.append(win_row)

    df_out = pd.DataFrame(data_matrix)
    return df_out, bold_cells


def generate_e0_all_variants_and_lambdas_table(
    results_dir: str,
    metric_prefix: str = "std_perf_auc",
) -> tuple[pd.DataFrame, dict]:
    """
    Generates a table comparing all 6 E(0) conditions across all environments:
    - E(0) [Oct 8, λ=1.0]
    - E(0) [Oct 8, λ=0.95]
    - E(0) [Oct 5, λ=1.0]
    - E(0) [Oct 5, λ=0.95]
    - E(0) [Sept 14, λ=1.0]
    - E(0) [Sept 14, λ=0.95]
    """
    import glob
    all_files = glob.glob(os.path.join(results_dir, "*", "summary_all_lrs.csv"))
    if not all_files:
        return pd.DataFrame(), {}

    dfs = [pd.read_csv(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True)

    e_df = df[df["algo"].isin(["e0_sep14", "e0_oct5", "e0_oct8"])].copy()
    if e_df.empty:
        return pd.DataFrame(), {}

    mean_col = f"{metric_prefix}_mean"
    sem_col = f"{metric_prefix}_sem"

    e_df["condition"] = e_df["algo"] + "_lam_" + e_df["return_lambda"].astype(str)
    best_runs = e_df.loc[e_df.groupby(["env", "condition"])[mean_col].idxmax()].copy()

    cond_order = [
        "e0_oct8_lam_1.0",
        "e0_oct8_lam_0.95",
        "e0_oct5_lam_1.0",
        "e0_oct5_lam_0.95",
        "e0_sep14_lam_1.0",
        "e0_sep14_lam_0.95",
    ]

    cond_display = {
        "e0_oct8_lam_1.0": "E(0) [Oct 8, λ=1.0]",
        "e0_oct8_lam_0.95": "E(0) [Oct 8, λ=0.95]",
        "e0_oct5_lam_1.0": "E(0) [Oct 5, λ=1.0]",
        "e0_oct5_lam_0.95": "E(0) [Oct 5, λ=0.95]",
        "e0_sep14_lam_1.0": "E(0) [Sept 14, λ=1.0]",
        "e0_sep14_lam_0.95": "E(0) [Sept 14, λ=0.95]",
    }

    envs_present = [e for e in ALL_ENVS if e in best_runs["env"].unique()]
    if not envs_present:
        return pd.DataFrame(), {}

    data_matrix = []
    bold_cells = {}
    win_counts = {c: 0.0 for c in cond_order}

    for row_idx, env in enumerate(envs_present):
        row = {"Environment": ENV_DISPLAY_NAMES.get(env, env)}
        env_sub = best_runs[best_runs["env"] == env]

        row_vals = {}
        row_sems = {}
        for c in cond_order:
            sub = env_sub[env_sub["condition"] == c]
            if not sub.empty:
                v = float(sub[mean_col].values[0])
                s = float(sub[sem_col].values[0]) if sem_col in sub.columns else 0.0
                row_vals[c] = v
                row_sems[c] = s
            else:
                row_vals[c] = np.nan
                row_sems[c] = 0.0

        valid_conds = [c for c in cond_order if not np.isnan(row_vals[c])]
        if valid_conds:
            best_c = max(valid_conds, key=lambda c: row_vals[c])
            best_val = row_vals[best_c]
            best_sem = row_sems[best_c]

            winners = []
            for c in valid_conds:
                val = row_vals[c]
                sem = row_sems[c]
                if (best_val - val) <= (best_sem + sem + 1e-6):
                    winners.append(c)

            for w in winners:
                win_counts[w] += 1.0 / len(winners)

            for col_idx, c in enumerate(cond_order):
                col_name = cond_display.get(c, c)
                v = row_vals[c]
                s = row_sems[c]
                if np.isnan(v):
                    row[col_name] = "-"
                else:
                    row[col_name] = f"{v:.3f} ± {s:.3f}"
                    if c in winners:
                        bold_cells[(row_idx, col_idx + 1)] = True
        else:
            for col_idx, c in enumerate(cond_order):
                col_name = cond_display.get(c, c)
                row[col_name] = "-"

        data_matrix.append(row)

    # Summary Row 1: Overall Mean Performance
    mean_row = {"Environment": "Mean Performance"}
    for c in cond_order:
        col_name = cond_display.get(c, c)
        sub = best_runs[best_runs["condition"] == c]
        if not sub.empty:
            vals = sub[mean_col].values
            avg = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            mean_row[col_name] = f"{avg:.3f} ± {sem:.3f}"
        else:
            mean_row[col_name] = "-"
    data_matrix.append(mean_row)

    # Summary Row 2: Envs Won
    win_row = {"Environment": f"Envs Won (out of {len(envs_present)})"}
    for c in cond_order:
        col_name = cond_display.get(c, c)
        win_row[col_name] = f"{win_counts[c]:.1f}/{len(envs_present)}"
    data_matrix.append(win_row)

    df_out = pd.DataFrame(data_matrix)
    return df_out, bold_cells


def plot_performance_heatmap_matrix(
    summary_df: pd.DataFrame,
    algo_keys: list[str],
    metric_prefix: str = "std_perf_auc",
    output_pdf: str = "auc_performance_by_env_heatmap.pdf",
    title: str = "AUC Standardized Performance Matrix",
):
    """
    Generates a publication-grade heatmap matrix plot:
    Rows = Environments, Columns = Algorithms.
    Each cell contains the numerical value (mean ± sem) and color-coded background.
    """
    mean_col = f"{metric_prefix}_mean"
    sem_col = f"{metric_prefix}_sem"

    envs_present = [e for e in ALL_ENVS if e in summary_df["env"].unique()]
    algos_present = [a for a in algo_keys if a in summary_df["algo"].unique()]

    if not envs_present or not algos_present:
        return

    matrix_mean = np.zeros((len(envs_present), len(algos_present)))
    matrix_sem = np.zeros((len(envs_present), len(algos_present)))
    text_labels = []

    for r_idx, env in enumerate(envs_present):
        row_labels = []
        env_sub = summary_df[summary_df["env"] == env]
        for c_idx, a in enumerate(algos_present):
            sub = env_sub[env_sub["algo"] == a]
            if not sub.empty:
                v = float(sub[mean_col].values[0])
                s = float(sub[sem_col].values[0]) if sem_col in sub.columns else 0.0
                matrix_mean[r_idx, c_idx] = v
                matrix_sem[r_idx, c_idx] = s
                row_labels.append(f"{v:.3f}\n±{s:.3f}")
            else:
                matrix_mean[r_idx, c_idx] = np.nan
                matrix_sem[r_idx, c_idx] = 0.0
                row_labels.append("-")
        text_labels.append(row_labels)

    fig_width = max(8.5, len(algos_present) * 2.0)
    fig_height = max(6.5, len(envs_present) * 0.75 + 1.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.cm.Blues
    im = ax.imshow(matrix_mean, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Ticks & labels
    ax.set_xticks(np.arange(len(algos_present)))
    col_labels = [ALGO_DISPLAY_NAMES.get(a, a) for a in algos_present]
    ax.set_xticklabels(col_labels, fontsize=10, rotation=20, ha="right", fontweight="bold")
    ax.set_yticks(np.arange(len(envs_present)))
    ax.set_yticklabels([ENV_DISPLAY_NAMES.get(e, e) for e in envs_present], fontsize=10, fontweight="bold")

    # Add text annotations in each cell
    for r in range(len(envs_present)):
        row_vals = matrix_mean[r, :]
        valid_indices = [c for c in range(len(algos_present)) if not np.isnan(row_vals[c])]
        best_c = max(valid_indices, key=lambda c: row_vals[c]) if valid_indices else -1
        best_v = row_vals[best_c] if best_c >= 0 else -1
        best_s = matrix_sem[r, best_c] if best_c >= 0 else 0

        for c in range(len(algos_present)):
            val = matrix_mean[r, c]
            txt = text_labels[r][c]
            if np.isnan(val):
                ax.text(c, r, "-", ha="center", va="center", color="#888888", fontsize=9.5)
            else:
                is_winner = (best_v - val) <= (best_s + matrix_sem[r, c] + 1e-6)
                text_color = "white" if val > 0.65 else "black"
                weight = "bold" if is_winner else "normal"
                if is_winner:
                    rect = plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, fill=False, edgecolor="#0b6623", linewidth=2.2)
                    ax.add_patch(rect)
                ax.text(c, r, txt, ha="center", va="center", color=text_color, fontsize=9.5, fontweight=weight)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Standardized Performance (0=Random, 1=Optimal)", fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    png_path = output_pdf.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] {output_pdf} and {png_path}")


def build_comparison_table(
    df: pd.DataFrame,
    algo_keys: list[str],
    table_title: str,
) -> pd.DataFrame:
    """Builds a formatted summary table comparing selected algorithms across environments."""
    sub_df = df[df["algo"].isin(algo_keys)].copy()
    if sub_df.empty:
        return pd.DataFrame()

    rows = []
    for algo_key in algo_keys:
        algo_sub = sub_df[sub_df["algo"] == algo_key]
        if algo_sub.empty:
            continue

        name = ALGO_DISPLAY_NAMES.get(algo_key, algo_key)

        mean_final_ret = algo_sub["std_perf_final_mean"].mean()
        sem_final_ret = algo_sub["std_perf_final_mean"].std(ddof=1) / np.sqrt(len(algo_sub)) if len(algo_sub) > 1 else 0.0

        mean_auc_ret = algo_sub["std_perf_auc_mean"].mean()
        sem_auc_ret = algo_sub["std_perf_auc_mean"].std(ddof=1) / np.sqrt(len(algo_sub)) if len(algo_sub) > 1 else 0.0

        mean_auc_ve = algo_sub["ve_auc_mean"].mean() if "ve_auc_mean" in algo_sub.columns else np.nan
        sem_auc_ve = algo_sub["ve_auc_mean"].std(ddof=1) / np.sqrt(len(algo_sub)) if len(algo_sub) > 1 and "ve_auc_mean" in algo_sub.columns else 0.0

        mean_auc_cos = algo_sub["cossim_auc_mean"].mean() if "cossim_auc_mean" in algo_sub.columns else np.nan
        sem_auc_cos = algo_sub["cossim_auc_mean"].std(ddof=1) / np.sqrt(len(algo_sub)) if len(algo_sub) > 1 and "cossim_auc_mean" in algo_sub.columns else 0.0

        rows.append({
            "algo_key": algo_key,
            "Algorithm": name,
            "Final Std Return": f"{mean_final_ret:.4f} ± {sem_final_ret:.4f}",
            "AUC Std Return": f"{mean_auc_ret:.4f} ± {sem_auc_ret:.4f}",
            "AUC Value Error": f"{mean_auc_ve:.4f} ± {sem_auc_ve:.4f}" if pd.notna(mean_auc_ve) else "N/A",
            "AUC Adv CosSim": f"{mean_auc_cos:.4f} ± {sem_auc_cos:.4f}" if pd.notna(mean_auc_cos) else "N/A",
        })

    # Compute per-environment wins and average rank
    env_list = sub_df["env"].unique()
    wins_ret = {a: 0 for a in algo_keys}
    ranks_ret = {a: [] for a in algo_keys}

    for env_name in env_list:
        e_df = sub_df[sub_df["env"] == env_name]
        if len(e_df) < len(algo_keys):
            continue
        scores = {}
        for a in algo_keys:
            r = e_df[e_df["algo"] == a]
            scores[a] = r["std_perf_auc_mean"].values[0] if not r.empty else -1e9

        # Best algo
        best_a = max(scores, key=scores.get)
        wins_ret[best_a] += 1

        # Ranks (1 = highest score)
        sorted_algos = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        for r_idx, a in enumerate(sorted_algos):
            ranks_ret[a].append(r_idx + 1)

    for r in rows:
        ak = r["algo_key"]
        w = wins_ret.get(ak, 0)
        n_e = len(env_list)
        pct = (w / n_e * 100) if n_e > 0 else 0
        r["Return Wins"] = f"{w}/{n_e} ({pct:.0f}%)"
        avg_rank = np.mean(ranks_ret.get(ak, [np.nan]))
        r["Mean Rank"] = f"{avg_rank:.2f}"
        del r["algo_key"]

    return pd.DataFrame(rows)


def build_pairwise_record_table(df: pd.DataFrame) -> pd.DataFrame:
    """Builds pairwise head-to-head records across key matchups."""
    available_algos = set(df["algo"].unique())

    oct8_cross = "e0_oct8_lam1" if "e0_oct8_lam1" in available_algos else "e0_oct8"
    oct5_cross = "e0_oct5_lam1" if "e0_oct5_lam1" in available_algos else "e0_oct5"
    oct8_cross_label = "E(0) [Oct 8, λ=1]" if "e0_oct8_lam1" in available_algos else "E(0) [Oct 8]"
    oct5_cross_label = "E(0) [Oct 5, λ=1]" if "e0_oct5_lam1" in available_algos else "E(0) [Oct 5]"

    matchups = [
        # Cross-algorithm comparisons (E constrained to lambda=1.0)
        (oct8_cross, "td0", f"{oct8_cross_label} vs TD(0)"),
        (oct8_cross, "mc", f"{oct8_cross_label} vs MC"),
        (oct5_cross, "td0", f"{oct5_cross_label} vs TD(0)"),
        (oct5_cross, "mc", f"{oct5_cross_label} vs MC"),
        ("td0", "mc", "TD(0) vs MC"),
        # E-variant comparisons (E tuned over both lambda=0.95 and lambda=1.0)
        ("e0_oct8", "e0_sep14", "E(0) [Oct 8] vs E(0) [Sept 14]"),
        ("e0_oct8", "e0_oct5", "E(0) [Oct 8] vs E(0) [Oct 5]"),
        ("e0_oct5", "e0_sep14", "E(0) [Oct 5] vs E(0) [Sept 14]"),
    ]

    metrics_to_eval = [
        ("std_perf_auc_mean", "std_perf_auc_sem", "AUC Std Return", True),
        ("std_perf_final_mean", "std_perf_final_sem", "Final Std Return", True),
        ("ve_auc_mean", "ve_auc_sem", "AUC Value Error", False),  # lower is better
        ("cossim_auc_mean", "cossim_auc_sem", "AUC Adv CosSim", True),
    ]

    env_list = df["env"].unique()
    rows = []

    for a1, a2, label in matchups:
        row_data = {"Head-to-Head Matchup": label}
        tot_w, tot_l, tot_t = 0, 0, 0

        for mean_col, sem_col, m_title, higher_better in metrics_to_eval:
            w, l, t = 0, 0, 0
            for env_name in env_list:
                e_df = df[df["env"] == env_name]
                r1 = e_df[e_df["algo"] == a1]
                r2 = e_df[e_df["algo"] == a2]

                if r1.empty or r2.empty or mean_col not in r1.columns or mean_col not in r2.columns:
                    continue

                v1 = r1[mean_col].values[0]
                v2 = r2[mean_col].values[0]
                s1 = r1[sem_col].values[0] if sem_col in r1.columns else 0.0
                s2 = r2[sem_col].values[0] if sem_col in r2.columns else 0.0

                if pd.isna(v1) or pd.isna(v2):
                    continue

                threshold = s1 + s2
                diff = (v1 - v2) if higher_better else (v2 - v1)

                if abs(v1 - v2) <= threshold:
                    t += 1
                elif diff > 0:
                    w += 1
                else:
                    l += 1

            pct = (w / (w + l + 1e-12)) * 100 if (w + l) > 0 else 0
            row_data[m_title] = f"{w}W - {l}L - {t}T ({pct:.0f}%)"
            tot_w += w
            tot_l += l
            tot_t += t

        tot_pct = (tot_w / (tot_w + tot_l + 1e-12)) * 100 if (tot_w + tot_l) > 0 else 0
        row_data["Overall Record"] = f"{tot_w}W - {tot_l}L - {tot_t}T ({tot_pct:.0f}%)"
        rows.append(row_data)

    return pd.DataFrame(rows)


def build_tuned_lrs_table(df: pd.DataFrame) -> pd.DataFrame:
    """Builds a table showing the selected critic learning rates and return lambdas per algorithm and env."""
    df_copy = df.copy()
    if "best_return_lambda" in df_copy.columns:
        df_copy["config_str"] = df_copy.apply(
            lambda r: f"lr={r['best_critic_lr']}" if r["algo"] in ["td0", "mc"] or pd.isna(r.get("best_return_lambda"))
            else f"lr={r['best_critic_lr']}, λ={r['best_return_lambda']}",
            axis=1,
        )
        val_col = "config_str"
    else:
        val_col = "best_critic_lr"

    piv = df_copy.pivot(index="env", columns="algo", values=val_col)
    piv = piv.reindex(columns=[a for a in ALGO_ORDER if a in piv.columns])
    piv = piv.rename(columns=ALGO_DISPLAY_NAMES)
    piv.index = [ENV_DISPLAY_NAMES.get(e, e) for e in piv.index]
    piv = piv.reset_index().rename(columns={"index": "Environment"})
    return piv


def main():
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else os.path.join(args.results_dir, "analysis_tables")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("Sampled PPO Sweep Publication Table Generator")
    print(f"Results Directory: {args.results_dir}")
    print(f"Output Directory:  {out_dir}")
    print("=" * 80)

    df = load_data(args.results_dir)
    available_algos = set(df["algo"].unique())

    oct8_cross = "e0_oct8_lam1" if "e0_oct8_lam1" in available_algos else "e0_oct8"
    oct5_cross = "e0_oct5_lam1" if "e0_oct5_lam1" in available_algos else "e0_oct5"
    sep14_cross = "e0_sep14_lam1" if "e0_sep14_lam1" in available_algos else "e0_sep14"

    # 1. Table 1: Core Comparison (MC vs TD(0) vs E(0) [Oct 8, lambda=1])
    t1 = build_comparison_table(
        df=df,
        algo_keys=["mc", "td0", oct8_cross],
        table_title="Table 1: Core Critic Comparison [Sampled PPO, k=32]",
    )
    if not t1.empty:
        save_table_outputs(t1, out_dir, "table_1_core_comparison", "Table 1: Core Critic Comparison [Sampled PPO, k=32]")

    # 2. Table 2: E(0) Variants Triad (Sept 14 vs Oct 5 vs Oct 8, tuned lambda in [0.95, 1.0])
    t2 = build_comparison_table(
        df=df,
        algo_keys=["e0_sep14", "e0_oct5", "e0_oct8"],
        table_title="Table 2: E(0) Sampling Formulations Triad [Sampled PPO, k=32]",
    )
    if not t2.empty:
        save_table_outputs(t2, out_dir, "table_2_e0_variants_triad", "Table 2: E(0) Sampling Formulations Triad [Sampled PPO, k=32]")

    # 3. Table 3: Full 5 Spectrum (Cross-comparison, E variants with lambda=1)
    t3 = build_comparison_table(
        df=df,
        algo_keys=["mc", "td0", sep14_cross, oct5_cross, oct8_cross],
        table_title="Table 3: Full 5-Algorithm Spectrum [Sampled PPO, k=32]",
    )
    if not t3.empty:
        save_table_outputs(t3, out_dir, "table_3_full_spectrum", "Table 3: Full 5-Algorithm Spectrum [Sampled PPO, k=32]")

    # 4. Table 4: Pairwise Head-to-Head Tournament Matrix
    t4 = build_pairwise_record_table(df)
    if not t4.empty:
        save_table_outputs(t4, out_dir, "table_4_pairwise_record", "Table 4: Pairwise Head-to-Head Decisive Record [Sampled PPO, k=32]")

    # 5. Table 5: Tuned Critic Learning Rates
    t5 = build_tuned_lrs_table(df)
    if not t5.empty:
        save_table_outputs(t5, out_dir, "table_5_tuned_critic_lrs", "Table 5: Tuned Critic Learning Rates [Sampled PPO, k=32]")

    # 6. Table 6: Area Under Curve (AUC) Standardized Policy Performance (Envs on rows, Algos on cols)
    full_cross_algos = ["mc", "td0", sep14_cross, oct5_cross, oct8_cross]
    t_auc_by_env, bold_auc_by_env = generate_env_by_algo_table(
        summary_df=df,
        algo_keys=full_cross_algos,
        metric_prefix="std_perf_auc",
    )
    if not t_auc_by_env.empty:
        save_table_outputs(
            t_auc_by_env,
            out_dir,
            "table_auc_performance_by_env",
            "Table 6: Area Under Curve (AUC) Standardized Policy Performance",
            bold_cells=bold_auc_by_env,
        )

    # 7. Table 7: Final Standardized Policy Performance (Envs on rows, Algos on cols)
    t_final_by_env, bold_final_by_env = generate_env_by_algo_table(
        summary_df=df,
        algo_keys=full_cross_algos,
        metric_prefix="std_perf_final",
    )
    if not t_final_by_env.empty:
        save_table_outputs(
            t_final_by_env,
            out_dir,
            "table_final_performance_by_env",
            "Table 7: Final Standardized Policy Performance (Random=0, Optimal=1)",
            bold_cells=bold_final_by_env,
        )

    # 8. Heatmap Matrix Plot: AUC Performance Matrix (Envs on rows, Algos on cols)
    plot_pdf = os.path.join(out_dir, "auc_performance_by_env_heatmap.pdf")
    plot_performance_heatmap_matrix(
        summary_df=df,
        algo_keys=full_cross_algos,
        metric_prefix="std_perf_auc",
        output_pdf=plot_pdf,
        title="Sampled PPO: Area Under Curve (AUC) Performance Matrix",
    )
    plots_dir = os.path.join(args.results_dir, "plots")
    if os.path.exists(plots_dir):
        plot_performance_heatmap_matrix(
            summary_df=df,
            algo_keys=full_cross_algos,
            metric_prefix="std_perf_auc",
            output_pdf=os.path.join(plots_dir, "auc_performance_by_env_heatmap.pdf"),
            title="Sampled PPO: Area Under Curve (AUC) Performance Matrix",
        )

    # 9. Table 8: All E(0) Sampling Formulations & Return Lambdas AUC Return Table
    t_e_all, bold_e_all = generate_e0_all_variants_and_lambdas_table(
        results_dir=args.results_dir,
        metric_prefix="std_perf_auc",
    )
    if not t_e_all.empty:
        save_table_outputs(
            t_e_all,
            out_dir,
            "table_e0_all_variants_and_lambdas_auc",
            "Table 8: E(0) Sampling Formulations & Return Lambdas AUC Standardized Return",
            bold_cells=bold_e_all,
        )

    print("\nAll publication tables and matrix plots successfully generated!")


if __name__ == "__main__":
    main()
