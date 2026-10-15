"""
scripts/sampled_fixed_policy_sweep/analyze_results.py

Comprehensive publication-focused data analysis for Sampled Fixed Policy Evaluation Sweeps.
Designed for ICML / NeurIPS reviewer standards:
1. Table 1 (Core Triad): Head-to-head comparison of MC, TD(0), and E(0)
   - Value Error (AUC and Final)
   - Advantage Cosine Similarity (Gradient alignment)
   - Normalized Greedy Return Score (0=Random, 1=Optimal)
   - Win counts across environments (e.g., 8/9 wins)
   - Average rank (1.0 = best, 3.0 = worst)
   - Generated for both bottleneck capacity (k=8) and high capacity (k=64)
2. Table 2 (Full Spectrum): Comparison across all 5 algorithms
   - MC, TD(0), E(0), TD(λ=0.8), E(λ=0.8)
   - Win counts and rank dominance
   - Generated for k=8 and k=64
3. Reviewer Diagnostic Insights:
   - Pairwise Head-to-Head W-L-T table (with statistical ties handled fairly: |v1 - v2| <= s1 + s2)
   - Selected Tuned Critic Learning Rates across algorithms and environments
4. Multi-Format Output:
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
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Sampled Fixed Policy Evaluation Results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/sampled_fixed_policy_sweep",
        help="Root results directory containing environment subfolders or summary_table_all.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for tables (default: <results_dir>/analysis_tables)",
    )
    return parser.parse_args()


def load_baselines(project_root: str) -> dict:
    json_path = os.path.join(project_root, "reference", "env_performance_baselines.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}


def load_data(results_dir: str, baselines: dict) -> pd.DataFrame:
    summary_path = os.path.join(results_dir, "summary_table_all.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
    else:
        dfs = []
        for env_name in ALL_ENVS:
            csv_path = os.path.join(results_dir, env_name, "summary_auc.csv")
            if os.path.exists(csv_path):
                dfs.append(pd.read_csv(csv_path))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError(f"No summary CSVs found in {results_dir}")

    if "env_name" in df.columns and "env" not in df.columns:
        df["env"] = df["env_name"]

    # Compute normalized greedy score if missing
    def calc_norm_return(row, col="final_greedy_perf_mean"):
        e = row["env"]
        if e in baselines and col in row and pd.notna(row[col]):
            v_star = float(baselines[e]["v_star"])
            v_rand = float(baselines[e]["v_rand"])
            d = v_star - v_rand
            if abs(d) > 1e-8:
                return (float(row[col]) - v_rand) / d
        return row[col] if col in row else np.nan

    def calc_norm_sem(row, col="final_greedy_perf_sem"):
        e = row["env"]
        if e in baselines and col in row and pd.notna(row[col]):
            v_star = float(baselines[e]["v_star"])
            v_rand = float(baselines[e]["v_rand"])
            d = v_star - v_rand
            if abs(d) > 1e-8:
                return float(row[col]) / d
        return row[col] if col in row else np.nan

    if "final_normalized_greedy_perf" not in df.columns:
        df["final_normalized_greedy_perf"] = df.apply(lambda r: calc_norm_return(r, "final_greedy_perf_mean"), axis=1)
    df["norm_greedy_sem"] = df.apply(lambda r: calc_norm_sem(r, "final_greedy_perf_sem"), axis=1)

    return df


def format_val(val: float, is_sci: bool = False, decimals: int = 4) -> str:
    if np.isnan(val):
        return "N/A"
    if is_sci or abs(val) < 1e-3 and val != 0:
        return f"{val:.2e}"
    return f"{val:.{decimals}f}"


def df_to_markdown(
    df: pd.DataFrame,
    title: str,
    bold_dict: dict[tuple[int, int], bool] | None = None,
) -> str:
    lines = [f"### {title}\n"]
    headers = [str(c) for c in df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row_idx, (_, row) in enumerate(df.iterrows()):
        row_str = []
        for col_idx, h in enumerate(df.columns):
            val_str = str(row[h])
            if bold_dict and (row_idx, col_idx) in bold_dict:
                val_str = f"**{val_str}**"
            row_str.append(val_str)
        lines.append("| " + " | ".join(row_str) + " |")

    return "\n".join(lines) + "\n"


def df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    bold_dict: dict[tuple[int, int], bool] | None = None,
) -> str:
    n_cols = len(df.columns)
    col_align = "l" + "r" * (n_cols - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_align}}}",
        r"\toprule",
    ]

    headers = [f"\\textbf{{{c}}}" for c in df.columns]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for row_idx, (_, row) in enumerate(df.iterrows()):
        row_str = []
        is_summary_row = any(term in str(row.iloc[0]).lower() for term in ["envs won", "mean rank", "average", "geom"])
        if is_summary_row and row_idx > 0 and not any(term in str(df.iloc[row_idx - 1, 0]).lower() for term in ["envs won", "mean rank", "average", "geom"]):
            lines.append(r"\midrule")

        for col_idx, h in enumerate(df.columns):
            val_str = str(row[h])
            val_str_tex = val_str.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
            if bold_dict and (row_idx, col_idx) in bold_dict:
                val_str_tex = f"\\textbf{{{val_str_tex}}}"
            if is_summary_row and col_idx == 0:
                val_str_tex = f"\\textit{{{val_str_tex}}}"
            row_str.append(val_str_tex)

        lines.append(" & ".join(row_str) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def render_table_to_pdf(
    df: pd.DataFrame,
    title: str,
    out_pdf_path: str,
    bold_dict: dict[tuple[int, int], bool] | None = None,
):
    n_rows, n_cols = df.shape
    fig_width = max(11.0, n_cols * 1.7)
    fig_height = max(4.0, (n_rows + 2) * 0.38 + 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    cell_text = []
    for r_idx in range(n_rows):
        row_vals = []
        for c_idx in range(n_cols):
            row_vals.append(str(df.iloc[r_idx, c_idx]))
        cell_text.append(row_vals)

    table = ax.table(
        cellText=cell_text,
        colLabels=list(df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.45)

    for c_idx in range(n_cols):
        cell = table[0, c_idx]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("#1a252f")

    for r_idx in range(n_rows):
        is_summary = any(term in str(df.iloc[r_idx, 0]).lower() for term in ["envs won", "mean rank", "average", "geom"])
        for c_idx in range(n_cols):
            cell = table[r_idx + 1, c_idx]
            cell.set_edgecolor("#dddddd")
            if c_idx == 0:
                cell.set_text_props(ha="left")
                cell.get_text().set_text("  " + cell.get_text().get_text())

            if is_summary:
                cell.set_facecolor("#eef2f7" if r_idx % 2 == 0 else "#f8f9fa")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#ffffff" if r_idx % 2 == 0 else "#fdfdfd")

            if bold_dict and (r_idx, c_idx) in bold_dict:
                cell.set_text_props(weight="bold")
                if not is_summary:
                    cell.set_facecolor("#e8f8f5")

    plt.title(title, fontsize=13, fontweight="bold", pad=20, color="#1a252f")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    plt.savefig(out_pdf_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_comparison_table(
    df: pd.DataFrame,
    algo_list: list[str],
    k_val: int,
    metrics: list[tuple[str, str, bool, str]],
) -> tuple[pd.DataFrame, dict[tuple[int, int], bool]]:
    sub_df = df[(df["k"] == k_val) & (df["algo"].isin(algo_list))].copy()
    envs_present = [e for e in ALL_ENVS if e in sub_df["env"].unique()]
    n_envs = len(envs_present)

    col_names = ["Environment"]
    for _, metric_name, _, _ in metrics:
        for algo in algo_list:
            col_names.append(f"{metric_name} ({ALGO_DISPLAY_NAMES.get(algo, algo)})")

    table_rows = []
    bold_dict = {}

    for r_idx, env_name in enumerate(envs_present):
        row = [ENV_DISPLAY_NAMES.get(env_name, env_name)]
        c_offset = 1

        for m_key, _, lower_better, fmt_type in metrics:
            algo_vals = {}
            for algo in algo_list:
                match = sub_df[(sub_df["env"] == env_name) & (sub_df["algo"] == algo)]
                if not match.empty and m_key in match.columns:
                    algo_vals[algo] = float(match[m_key].values[0])
                else:
                    algo_vals[algo] = np.nan

            valid_vals = {a: v for a, v in algo_vals.items() if not np.isnan(v)}
            best_algo = (min(valid_vals, key=valid_vals.get) if lower_better else max(valid_vals, key=valid_vals.get)) if valid_vals else None

            for i, algo in enumerate(algo_list):
                val = algo_vals[algo]
                dec = 4 if fmt_type != "pct" else 3
                row.append(format_val(val, is_sci=(fmt_type == "sci"), decimals=dec))
                if algo == best_algo:
                    bold_dict[(r_idx, c_offset + i)] = True

            c_offset += len(algo_list)
        table_rows.append(row)

    # Summary Row 1: Envs Won (out of N)
    win_row = [f"Envs Won (out of {n_envs})"]
    c_offset = 1
    for m_key, _, lower_better, _ in metrics:
        algo_wins = {a: 0 for a in algo_list}
        for env_name in envs_present:
            e_vals = {}
            for algo in algo_list:
                match = sub_df[(sub_df["env"] == env_name) & (sub_df["algo"] == algo)]
                if not match.empty and m_key in match.columns:
                    e_vals[algo] = float(match[m_key].values[0])
            valid = {a: v for a, v in e_vals.items() if not np.isnan(v)}
            if valid:
                w = min(valid, key=valid.get) if lower_better else max(valid, key=valid.get)
                algo_wins[w] += 1

        max_wins = max(algo_wins.values()) if algo_wins else 0
        for i, algo in enumerate(algo_list):
            win_row.append(f"{algo_wins[algo]}/{n_envs}")
            if algo_wins[algo] == max_wins and max_wins > 0:
                bold_dict[(len(table_rows), c_offset + i)] = True
        c_offset += len(algo_list)
    table_rows.append(win_row)

    # Summary Row 2: Average Rank
    rank_row = ["Average Rank (1.0 = best)"]
    c_offset = 1
    for m_key, _, lower_better, _ in metrics:
        algo_ranks = {a: [] for a in algo_list}
        for env_name in envs_present:
            e_vals = {}
            for algo in algo_list:
                match = sub_df[(sub_df["env"] == env_name) & (sub_df["algo"] == algo)]
                if not match.empty and m_key in match.columns:
                    e_vals[algo] = float(match[m_key].values[0])
            valid = {a: v for a, v in e_vals.items() if not np.isnan(v)}
            if len(valid) == len(algo_list):
                sorted_algos = sorted(valid, key=valid.get, reverse=not lower_better)
                for rank_idx, algo in enumerate(sorted_algos):
                    algo_ranks[algo].append(rank_idx + 1)

        mean_ranks = {a: (np.mean(algo_ranks[a]) if algo_ranks[a] else np.nan) for a in algo_list}
        best_rank = min(mean_ranks.values()) if any(not np.isnan(v) for v in mean_ranks.values()) else np.nan
        for i, algo in enumerate(algo_list):
            r_val = mean_ranks[algo]
            rank_row.append(f"{r_val:.2f}" if not np.isnan(r_val) else "N/A")
            if abs(r_val - best_rank) < 1e-4:
                bold_dict[(len(table_rows), c_offset + i)] = True
        c_offset += len(algo_list)
    table_rows.append(rank_row)

    # Summary Row 3: Mean across Envs
    mean_row = ["Mean across Envs"]
    c_offset = 1
    for m_key, _, lower_better, fmt_type in metrics:
        algo_means = {}
        for algo in algo_list:
            match = sub_df[sub_df["algo"] == algo]
            if not match.empty and m_key in match.columns:
                algo_means[algo] = float(match[m_key].dropna().mean())
            else:
                algo_means[algo] = np.nan

        valid_means = {a: v for a, v in algo_means.items() if not np.isnan(v)}
        best_mean = (min(valid_means, key=valid_means.get) if lower_better else max(valid_means, key=valid_means.get)) if valid_means else None

        for i, algo in enumerate(algo_list):
            val = algo_means[algo]
            dec = 4 if fmt_type != "pct" else 3
            mean_row.append(format_val(val, is_sci=(fmt_type == "sci"), decimals=dec))
            if algo == best_mean:
                bold_dict[(len(table_rows), c_offset + i)] = True
        c_offset += len(algo_list)
    table_rows.append(mean_row)

    res_df = pd.DataFrame(table_rows, columns=col_names)
    return res_df, bold_dict


def generate_pairwise_head_to_head_table(
    df: pd.DataFrame,
    matchups: list[tuple[str, str, str]],
    k_val: int,
    metrics: list[tuple[str, str, str, bool, str]],
) -> tuple[pd.DataFrame, dict[tuple[int, int], bool]]:
    """
    Generates a pairwise head-to-head table with explicit Wins, Losses, and Statistical Ties.
    For each matchup A vs B on each environment:
      - If |val_A - val_B| <= sem_A + sem_B: Tie (T).
      - Else if A is better: Win for A (W).
      - Else: Loss for A (L).
    """
    sub_df = df[df["k"] == k_val].copy()
    envs = [e for e in ALL_ENVS if e in sub_df["env"].unique()]

    col_names = ["Head-to-Head Matchup"] + [m[2] for m in metrics] + ["Overall Record"]
    table_rows = []
    bold_dict = {}

    for r_idx, (a1, a2, m_label) in enumerate(matchups):
        row = [m_label]
        total_w, total_l, total_t = 0, 0, 0

        for c_idx, (m_val, m_sem, m_name, lower_better, _) in enumerate(metrics, start=1):
            w, l, t = 0, 0, 0
            for e in envs:
                rows_1 = sub_df[(sub_df["env"] == e) & (sub_df["algo"] == a1)]
                rows_2 = sub_df[(sub_df["env"] == e) & (sub_df["algo"] == a2)]
                if rows_1.empty or rows_2.empty:
                    continue
                r1 = rows_1.iloc[0]
                r2 = rows_2.iloc[0]
                v1 = r1[m_val]
                v2 = r2[m_val]
                s1 = r1[m_sem] if m_sem in r1 and pd.notna(r1[m_sem]) else 0.0
                s2 = r2[m_sem] if m_sem in r2 and pd.notna(r2[m_sem]) else 0.0

                if np.isnan(v1) or np.isnan(v2):
                    continue

                diff = v1 - v2
                threshold = s1 + s2
                if abs(diff) <= threshold:
                    t += 1
                elif (lower_better and diff < 0) or (not lower_better and diff > 0):
                    w += 1
                else:
                    l += 1

            total_decisive = w + l
            eff_pct = (w / total_decisive * 100.0) if total_decisive > 0 else 50.0
            row.append(f"{w}W - {l}L - {t}T ({eff_pct:.0f}%)")
            total_w += w
            total_l += l
            total_t += t

            if w > l:
                bold_dict[(r_idx, c_idx)] = True

        overall_decisive = total_w + total_l
        overall_pct = (total_w / overall_decisive * 100.0) if overall_decisive > 0 else 50.0
        row.append(f"{total_w}W - {total_l}L - {total_t}T ({overall_pct:.0f}%)")
        if total_w > total_l:
            bold_dict[(r_idx, len(metrics) + 1)] = True

        table_rows.append(row)

    return pd.DataFrame(table_rows, columns=col_names), bold_dict


def generate_lr_selection_table(df: pd.DataFrame, k_val: int) -> tuple[pd.DataFrame, dict[tuple[int, int], bool]]:
    """Tabulates the selected tuned critic learning rate for each algorithm across environments."""
    sub_df = df[df["k"] == k_val].copy()
    envs = [e for e in ALL_ENVS if e in sub_df["env"].unique()]

    col_names = ["Environment"] + [ALGO_DISPLAY_NAMES.get(a, a) for a in ALGO_ORDER]
    table_rows = []

    for env_name in envs:
        row = [ENV_DISPLAY_NAMES.get(env_name, env_name)]
        for algo in ALGO_ORDER:
            m = sub_df[(sub_df["env"] == env_name) & (sub_df["algo"] == algo)]
            if not m.empty and "best_critic_lr" in m.columns:
                row.append(str(m["best_critic_lr"].values[0]))
            else:
                row.append("N/A")
        table_rows.append(row)

    return pd.DataFrame(table_rows, columns=col_names), {}


def save_table_suite(
    df: pd.DataFrame,
    base_name: str,
    title: str,
    caption: str,
    label: str,
    out_dir: str,
    bold_dict: dict[tuple[int, int], bool] | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    md_path = os.path.join(out_dir, f"{base_name}.md")
    tex_path = os.path.join(out_dir, f"{base_name}.tex")
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")

    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df_to_markdown(df, title, bold_dict))
    with open(tex_path, "w") as f:
        f.write(df_to_latex(df, caption, label, bold_dict))
    render_table_to_pdf(df, title, pdf_path, bold_dict)

    print(f"  [OK] Exported suite: {base_name} (CSV, MD, TeX, PDF)")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir if args.out_dir else os.path.join(results_dir, "analysis_tables"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    print("=" * 80)
    print("SAMPLED FIXED POLICY SWEEP - ICML / NEURIPS ANALYSIS TABLES")
    print(f"Results Directory: {results_dir}")
    print(f"Output Directory:  {out_dir}")
    print("=" * 80)

    baselines = load_baselines(project_root)
    df = load_data(results_dir, baselines)

    k_values = sorted(df["k"].unique()) if "k" in df.columns else [8]

    core_algos = ["mc", "td0", "e0_oct8"]
    e0_variants = ["e0_sep14", "e0_oct5", "e0_oct8"]
    full_algos = ["mc", "td0", "e0_sep14", "e0_oct5", "e0_oct8"]

    triad_metrics = [
        ("auc_weighted_ve_mean", "AUC Value Error", True, "sci"),
        ("auc_advantage_cossim_mean", "AUC Adv CosSim", False, "std"),
        ("final_normalized_greedy_perf", "Norm Return Score", False, "pct"),
    ]

    pairwise_matchups = [
        ("e0_oct8", "td0", "E(0) [Oct 8] vs TD(0)"),
        ("e0_oct8", "mc", "E(0) [Oct 8] vs MC"),
        ("e0_oct8", "e0_sep14", "E(0) [Oct 8] vs E(0) [Sept 14]"),
        ("e0_oct8", "e0_oct5", "E(0) [Oct 8] vs E(0) [Oct 5]"),
        ("e0_oct5", "e0_sep14", "E(0) [Oct 5] vs E(0) [Sept 14]"),
        ("td0", "mc", "TD(0) vs MC"),
    ]

    pairwise_metrics = [
        ("auc_weighted_ve_mean", "auc_weighted_ve_sem", "AUC Value Error", True, "sci"),
        ("auc_advantage_cossim_mean", "auc_advantage_cossim_sem", "AUC Adv CosSim", False, "std"),
        ("final_normalized_greedy_perf", "norm_greedy_sem", "Final Norm Return", False, "pct"),
    ]

    for k in k_values:
        print(f"\n--- Generating Analysis Tables for k={k} ---")

        # Table 1: Core Comparison (MC vs TD(0) vs E(0) [Oct 8])
        t1_df, t1_bold = generate_comparison_table(
            df=df,
            algo_list=core_algos,
            k_val=k,
            metrics=triad_metrics,
        )
        save_table_suite(
            t1_df,
            base_name=f"table1_core_comparison_k{k}",
            title=f"Table 1: Core Comparison (MC vs TD(0) vs E(0) [Oct 8]) [Sampled, k={k}]",
            caption=f"Head-to-head comparison of MC, TD(0), and E(0) [Start-Anchored Oct 8] under sampled rollouts (k={k}).",
            label=f"tab:sampled_core_k{k}",
            out_dir=out_dir,
            bold_dict=t1_bold,
        )

        # Table 2: E(0) Variants Triad (Sept 14 vs Oct 5 vs Oct 8)
        t2_df, t2_bold = generate_comparison_table(
            df=df,
            algo_list=e0_variants,
            k_val=k,
            metrics=triad_metrics,
        )
        save_table_suite(
            t2_df,
            base_name=f"table2_e0_variants_k{k}",
            title=f"Table 2: E(0) Formulations Comparison (Sept 14 vs Oct 5 vs Oct 8) [Sampled, k={k}]",
            caption=f"Comparison across the three sampled E(0) formulations under rollouts (k={k}).",
            label=f"tab:sampled_e0_variants_k{k}",
            out_dir=out_dir,
            bold_dict=t2_bold,
        )

        # Table 3: Full Spectrum (5 Algorithms)
        t3_df, t3_bold = generate_comparison_table(
            df=df,
            algo_list=full_algos,
            k_val=k,
            metrics=triad_metrics,
        )
        save_table_suite(
            t3_df,
            base_name=f"table3_full_spectrum_k{k}",
            title=f"Table 3: Full Spectrum (5 Algorithms) [Sampled, k={k}]",
            caption=f"Comprehensive evaluation across all 5 algorithms under sampled rollouts (k={k}).",
            label=f"tab:sampled_full_k{k}",
            out_dir=out_dir,
            bold_dict=t3_bold,
        )

        # Table 4: Pairwise Head-to-Head Tournament Matrix
        t4_df, t4_bold = generate_pairwise_head_to_head_table(
            df=df,
            matchups=pairwise_matchups,
            k_val=k,
            metrics=pairwise_metrics,
        )
        save_table_suite(
            t4_df,
            base_name=f"table4_pairwise_head_to_head_k{k}",
            title=f"Table 4: Pairwise Head-to-Head Decisive Record [Sampled, k={k}]",
            caption=f"Decisive head-to-head records (Wins - Losses - Statistical Ties) across environments (k={k}).",
            label=f"tab:sampled_h2h_k{k}",
            out_dir=out_dir,
            bold_dict=t4_bold,
        )

        # Table 5: Selected Tuned Critic LRs
        t5_df, t5_bold = generate_lr_selection_table(df=df, k_val=k)
        save_table_suite(
            t5_df,
            base_name=f"table5_selected_lrs_k{k}",
            title=f"Table 5: Selected Tuned Critic Learning Rates [Sampled, k={k}]",
            caption=f"Selected optimal critic learning rate for each algorithm across environments (k={k}).",
            label=f"tab:sampled_lrs_k{k}",
            out_dir=out_dir,
            bold_dict=t5_bold,
        )

    print(f"\nAll publication tables successfully created and exported to: {out_dir}")


if __name__ == "__main__":
    main()
