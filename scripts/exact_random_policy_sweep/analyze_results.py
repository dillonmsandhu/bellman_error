"""
scripts/exact_random_policy_sweep/analyze_results.py

Comprehensive publication-focused data analysis for Exact Policy Evaluation Sweeps.
Designed for ICML / NeurIPS reviewer standards:
1. Table 1 (Core Triad): Head-to-head comparison of TD(0), MC, and E(0)
   - Value Error (AUC and Final)
   - Normalized V_start Greedy Return Score (0=Random, 1=Optimal)
   - Advantage Cosine Similarity (Gradient alignment)
   - Win counts across environments (e.g., 8/9 wins)
   - Average rank (1.0 = best, 3.0 = worst)
   - Generated for both bottleneck capacity (k=8) and high capacity (k=64)
2. Table 2 (Full Spectrum): Comparison across all 5 algorithms
   - MC, TD(0), E(0), TD(λ=0.8), E(λ=0.8)
   - Win counts and rank dominance
   - Generated for k=8 and k=64
3. Reviewer Diagnostic Insights:
   - Pairwise Tournament Matrix (5x5 head-to-head wins)
   - Relative Improvement Ratios (E(0) vs TD(0), E(λ) vs TD(λ))
   - Capacity Scaling Effect (k=8 degradation vs k=64)
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
    "fourrooms-dense": "FourRooms (Dense)",
    "fourrooms-mines": "FourRooms + Mines",
    "fourrooms-mines-dense": "FourRooms + Mines (Dense)",
    "eightrooms-misc": "EightRooms (Sparse)",
    "eightrooms-dense": "EightRooms (Dense)",
    "whirlpool-misc": "Whirlpool (Irreversible)",
    "SpaceInvadersExactValue": "Space Invaders",
    "mountaincar-dense": "Mountain Car (Dense)",
}

ALGO_ORDER = ["mc", "td0", "e0", "td_lambda", "e_lambda"]

ALGO_DISPLAY_NAMES = {
    "mc": "MC (Exact)",
    "td0": "TD(0) [Exact]",
    "e0": "E(0) [Exact]",
    "td_lambda": "TD(λ=0.8) [Exact]",
    "e_lambda": "E(λ=0.8) [Exact]",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Exact Policy Evaluation Results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/exact_random_policy_sweep",
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
    print(f"[Warning] Baselines JSON not found at {json_path}")
    return {}


def load_dataset(results_dir: str, baselines: dict) -> pd.DataFrame:
    """Loads all summary_auc.csv files or summary_table_all.csv and computes normalized metrics."""
    summary_all_path = os.path.join(results_dir, "summary_table_all.csv")
    dfs = []

    if os.path.exists(summary_all_path):
        try:
            dfs.append(pd.read_csv(summary_all_path))
        except Exception as e:
            print(f"[Warning] Failed to load {summary_all_path}: {e}")

    if not dfs:
        # Search in environment subdirectories
        for env_name in ALL_ENVS:
            csv_path = os.path.join(results_dir, env_name, "summary_auc.csv")
            if os.path.exists(csv_path):
                try:
                    dfs.append(pd.read_csv(csv_path))
                except Exception as e:
                    print(f"[Warning] Failed to load {csv_path}: {e}")

    if not dfs:
        raise FileNotFoundError(f"No summary data found in {results_dir}")

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["env", "k", "algo"])

    import pickle
    # Populate auc_greedy_perf_sem from out.pkl if missing
    if "auc_greedy_perf_sem" not in df.columns or df["auc_greedy_perf_sem"].isna().any():
        df["auc_greedy_perf_sem"] = np.nan
        for idx, row in df.iterrows():
            e = row["env"]
            k = int(row["k"])
            a = row["algo"]
            lr = float(row["best_lr"])
            pkl_p = os.path.join(results_dir, e, "out.pkl")
            if os.path.exists(pkl_p):
                try:
                    with open(pkl_p, "rb") as f:
                        d = pickle.load(f)
                    lrs_data = d.get("data", {}).get(k, {}).get(a, {})
                    matched_lr = None
                    for l_k in lrs_data:
                        if abs(float(l_k) - lr) < 1e-6:
                            matched_lr = l_k
                            break
                    if matched_lr is not None and "nn_greedy_performance" in lrs_data[matched_lr]:
                        gp = lrs_data[matched_lr]["nn_greedy_performance"]
                        auc_seeds = np.mean(gp, axis=1)
                        sem = float(np.std(auc_seeds) / np.sqrt(max(1, len(auc_seeds))))
                        df.loc[idx, "auc_greedy_perf_sem"] = sem
                except Exception:
                    pass

    # Compute normalized score for greedy return: (V - V_rand) / (V_star - V_rand)
    def calc_norm_return(row, col="auc_greedy_perf"):
        e = row["env"]
        if e in baselines and col in row:
            v_star = baselines[e]["v_star"]
            v_rand = baselines[e]["v_rand"]
            d = v_star - v_rand
            if abs(d) > 1e-8:
                return (float(row[col]) - v_rand) / d
        return row[col] if col in row else np.nan

    def calc_norm_sem(row, col="auc_greedy_perf_sem"):
        e = row["env"]
        if e in baselines and col in row and pd.notna(row[col]):
            v_star = baselines[e]["v_star"]
            v_rand = baselines[e]["v_rand"]
            d = v_star - v_rand
            if abs(d) > 1e-8:
                return float(row[col]) / d
        return row[col] if col in row else np.nan

    df["norm_vstart_auc"] = df.apply(lambda r: calc_norm_return(r, "auc_greedy_perf"), axis=1)
    df["norm_vstart_auc_sem"] = df.apply(lambda r: calc_norm_sem(r, "auc_greedy_perf_sem"), axis=1)
    if "final_greedy_perf" in df.columns:
        df["norm_vstart_final"] = df.apply(lambda r: calc_norm_return(r, "final_greedy_perf"), axis=1)

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
    """Renders a pandas DataFrame as a GitHub-Flavored Markdown table with optional cell bolding."""
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
    """Renders a pandas DataFrame as a publication-ready LaTeX table using booktabs."""
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

    # Header
    headers = [f"\\textbf{{{c}}}" for c in df.columns]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    # Body
    for row_idx, (_, row) in enumerate(df.iterrows()):
        row_str = []
        is_summary_row = any(term in str(row.iloc[0]).lower() for term in ["envs won", "mean rank", "average", "geom"])
        if is_summary_row and row_idx > 0 and not any(term in str(df.iloc[row_idx - 1, 0]).lower() for term in ["envs won", "mean rank", "average", "geom"]):
            lines.append(r"\midrule")

        for col_idx, h in enumerate(df.columns):
            val_str = str(row[h])
            # Escape LaTeX special chars
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
    """Renders a DataFrame as an publication-ready vector PDF table using Matplotlib."""
    n_rows, n_cols = df.shape
    fig_width = max(11.0, n_cols * 1.7)
    fig_height = max(4.0, (n_rows + 2) * 0.38 + 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    cell_text = []
    for r_idx in range(n_rows):
        row_vals = []
        for c_idx in range(n_cols):
            val = str(df.iloc[r_idx, c_idx])
            row_vals.append(val)
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

    # Style header and rows
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
            # Left align environment name column
            if c_idx == 0:
                cell.set_text_props(ha="left")
                cell.get_text().set_text("  " + cell.get_text().get_text())

            if is_summary:
                cell.set_facecolor("#eef2f7" if r_idx % 2 == 0 else "#f8f9fa")
                cell.set_text_props(weight="bold")
            else:
                if r_idx % 2 == 1:
                    cell.set_facecolor("#f9f9f9")
                else:
                    cell.set_facecolor("#ffffff")

            if bold_dict and (r_idx, c_idx) in bold_dict:
                cell.set_text_props(weight="bold", color="#1b4f72")
                cell.set_facecolor("#d4efdf" if not is_summary else "#c8e6c9")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=16, color="#222222")
    plt.tight_layout()
    plt.savefig(out_pdf_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_comparison_table(
    df: pd.DataFrame,
    algo_list: list[str],
    k_val: int,
    metrics: list[tuple[str, str, bool, str]],  # (metric_key, display_name, lower_is_better, format_type)
) -> tuple[pd.DataFrame, dict[tuple[int, int], bool]]:
    """
    Builds a unified comparison table comparing selected algorithms for a given k.
    Includes per-environment rows and summary rows:
    - Envs Won (out of N)
    - Mean Rank
    - Mean / Geometric Mean
    """
    sub_df = df[(df["k"] == k_val) & (df["algo"].isin(algo_list))].copy()
    envs_present = [e for e in ALL_ENVS if e in sub_df["env"].unique()]
    n_envs = len(envs_present)

    # Columns: Environment, then for each metric: [algo1, algo2, ...]
    col_names = ["Environment"]
    for _, metric_name, _, _ in metrics:
        for algo in algo_list:
            col_names.append(f"{metric_name} ({ALGO_DISPLAY_NAMES.get(algo, algo)})")

    table_rows = []
    bold_dict = {}

    # 1. Environment rows
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
            if valid_vals:
                best_algo = min(valid_vals, key=valid_vals.get) if lower_better else max(valid_vals, key=valid_vals.get)
            else:
                best_algo = None

            for i, algo in enumerate(algo_list):
                val = algo_vals[algo]
                is_sci = (fmt_type == "sci")
                dec = 4 if fmt_type != "pct" else 3
                val_formatted = format_val(val, is_sci=is_sci, decimals=dec)
                row.append(val_formatted)

                if algo == best_algo:
                    bold_dict[(r_idx, c_offset + i)] = True

            c_offset += len(algo_list)
        table_rows.append(row)

    # 2. Win Counts row
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

    # 3. Mean Rank row (1.0 = best)
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

    # 4. Aggregate Mean row
    mean_row = ["Mean across Envs"]
    c_offset = 1
    for m_key, _, lower_better, fmt_type in metrics:
        algo_means = {}
        for algo in algo_list:
            vals = sub_df[sub_df["algo"] == algo][m_key].dropna().values
            algo_means[algo] = float(np.mean(vals)) if len(vals) > 0 else np.nan

        valid_means = {a: v for a, v in algo_means.items() if not np.isnan(v)}
        best_mean = min(valid_means, key=valid_means.get) if lower_better else max(valid_means, key=valid_means.get) if valid_means else None

        for i, algo in enumerate(algo_list):
            m_val = algo_means[algo]
            dec = 4 if fmt_type != "pct" else 3
            mean_row.append(format_val(m_val, is_sci=(fmt_type == "sci"), decimals=dec))
            if algo == best_mean:
                bold_dict[(len(table_rows), c_offset + i)] = True
        c_offset += len(algo_list)
    table_rows.append(mean_row)

    res_df = pd.DataFrame(table_rows, columns=col_names)
    return res_df, bold_dict


def generate_pairwise_tournament_matrix(df: pd.DataFrame, k_val: int) -> tuple[pd.DataFrame, dict]:
    """Generates a 5x5 pairwise head-to-head tournament matrix for Value Error AUC."""
    sub_df = df[df["k"] == k_val]
    piv = sub_df.pivot(index="env", columns="algo", values="auc_ve_mean")
    algos = [a for a in ALGO_ORDER if a in piv.columns]
    n_envs = len(piv.dropna())

    matrix_rows = []
    bold_dict = {}

    for r_idx, a1 in enumerate(algos):
        row = [ALGO_DISPLAY_NAMES.get(a1, a1)]
        wins_total = 0
        comparisons_total = 0
        for c_idx, a2 in enumerate(algos):
            if a1 == a2:
                row.append("—")
            else:
                wins = (piv[a1] < piv[a2]).sum()
                losses = (piv[a1] > piv[a2]).sum()
                row.append(f"{wins}-{losses}")
                wins_total += wins
                comparisons_total += (wins + losses)
                if wins > losses:
                    bold_dict[(r_idx, c_idx + 1)] = True
        win_rate = (wins_total / comparisons_total * 100) if comparisons_total > 0 else 0
        row.append(f"{win_rate:.1f}%")
        matrix_rows.append(row)

    cols = ["Algorithm"] + [ALGO_DISPLAY_NAMES.get(a, a) for a in algos] + ["Win Rate (%)"]
    return pd.DataFrame(matrix_rows, columns=cols), bold_dict


def generate_env_win_rate_table(
    df: pd.DataFrame,
    algo_list: list[str],
    k_val: int,
    metrics: list[tuple[str, str, str, bool, str]],
) -> tuple[pd.DataFrame, dict[tuple[int, int], bool]]:
    """
    Generates an environment win rate table accounting for statistical ties.
    For each metric and each environment:
      - The algorithm with the top mean is identified.
      - Any algorithm whose performance overlaps within 1-SEM uncertainty
        (|val - best_val| <= sem + best_sem) is classified as statistically tied for 1st place.
      - If m algorithms tie for 1st place, each receives 1/m fractional win credit.
    Each cell reports the effective win rate % and the breakdown:
      e.g. '44.4% (4.0/9) [3w+2t]' (3 outright wins, 2 shared ties).
    """
    sub_df = df[df["k"] == k_val].copy()
    envs = [e for e in ALL_ENVS if e in sub_df["env"].unique()]
    n_envs = len(envs)

    col_names = ["Algorithm"] + [m[2] for m in metrics] + ["Overall Mean Win Rate"]

    # Track fractional wins, outright wins, and ties:
    frac_wins = {a: {m[2]: 0.0 for m in metrics} for a in algo_list}
    outright_wins = {a: {m[2]: 0 for m in metrics} for a in algo_list}
    tie_counts = {a: {m[2]: 0 for m in metrics} for a in algo_list}

    for m_val, m_sem, m_name, lower_better, _ in metrics:
        for e in envs:
            e_rows = sub_df[(sub_df["env"] == e) & (sub_df["algo"].isin(algo_list))]
            vals = {r["algo"]: r[m_val] for _, r in e_rows.iterrows() if pd.notna(r[m_val])}
            sems = {r["algo"]: (r[m_sem] if m_sem in r and pd.notna(r[m_sem]) else 0.0) for _, r in e_rows.iterrows() if pd.notna(r[m_val])}

            if not vals:
                continue

            best_algo = min(vals, key=vals.get) if lower_better else max(vals, key=vals.get)
            best_val = vals[best_algo]
            best_sem = sems[best_algo]

            # Find all tied with best within 1-SEM: |val - best_val| <= (sem + best_sem)
            tied = []
            for a in algo_list:
                if a not in vals:
                    continue
                diff = abs(vals[a] - best_val)
                unc = sems[a] + best_sem
                if diff <= unc or diff < 1e-5:
                    tied.append(a)

            if len(tied) == 1:
                frac_wins[best_algo][m_name] += 1.0
                outright_wins[best_algo][m_name] += 1
            else:
                for a in tied:
                    frac_wins[a][m_name] += 1.0 / len(tied)
                    tie_counts[a][m_name] += 1

    table_rows = []
    col_pcts = {m[2]: [] for m in metrics}
    col_pcts["Overall Mean Win Rate"] = []

    total_possible = n_envs * len(metrics)
    for a in algo_list:
        row = [ALGO_DISPLAY_NAMES.get(a, a)]
        total_a_frac = 0.0
        for _, _, m_name, _, _ in metrics:
            fw = frac_wins[a][m_name]
            ow = outright_wins[a][m_name]
            tc = tie_counts[a][m_name]
            pct = (fw / n_envs * 100.0) if n_envs > 0 else 0.0
            row.append(f"{pct:.1f}% ({fw:.1f}/9) [{ow}w+{tc}t]")
            col_pcts[m_name].append(pct)
            total_a_frac += fw

        overall_pct = (total_a_frac / total_possible * 100.0) if total_possible > 0 else 0.0
        row.append(f"{overall_pct:.1f}% ({total_a_frac:.1f}/{total_possible})")
        col_pcts["Overall Mean Win Rate"].append(overall_pct)
        table_rows.append(row)

    bold_dict = {}
    for col_idx, col_header in enumerate(col_names[1:], start=1):
        pcts = col_pcts[col_header]
        max_pct = max(pcts) if pcts else 0
        for row_idx, pct in enumerate(pcts):
            if abs(pct - max_pct) < 1e-4 and max_pct > 0:
                bold_dict[(row_idx, col_idx)] = True

    return pd.DataFrame(table_rows, columns=col_names), bold_dict


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
    Cells report: '{W}W - {L}L - {T}T ({eff_pct:.0f}%)'.
    Bold is applied to matchups with a winning decisive record (W > L).
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

                diff = v1 - v2 if lower_better else v2 - v1
                unc = s1 + s2
                if abs(diff) <= unc or abs(diff) < 1e-5:
                    t += 1
                elif diff < 0:
                    w += 1
                else:
                    l += 1

            eff_pct = ((w + 0.5 * t) / 9.0 * 100.0) if len(envs) > 0 else 0.0
            row.append(f"{w}W - {l}L - {t}T ({eff_pct:.0f}%)")
            if w > l:
                bold_dict[(r_idx, c_idx)] = True

            total_w += w
            total_l += l
            total_t += t

        total_poss = total_w + total_l + total_t
        tot_eff = ((total_w + 0.5 * total_t) / total_poss * 100.0) if total_poss > 0 else 0.0
        row.append(f"{total_w}W - {total_l}L - {total_t}T ({tot_eff:.1f}%)")
        if total_w > total_l:
            bold_dict[(r_idx, len(col_names) - 1)] = True

        table_rows.append(row)

    return pd.DataFrame(table_rows, columns=col_names), bold_dict


def save_table_bundle(
    df: pd.DataFrame,
    bold_dict: dict,
    base_name: str,
    title: str,
    caption: str,
    out_dir: str,
):
    """Exports a table into CSV, Markdown, LaTeX, and rendered vector PDF."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. CSV
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    df.to_csv(csv_path, index=False)

    # 2. Markdown
    md_content = df_to_markdown(df, title, bold_dict)
    md_path = os.path.join(out_dir, f"{base_name}.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    # 3. LaTeX
    tex_content = df_to_latex(df, caption, f"tab:{base_name}", bold_dict)
    tex_path = os.path.join(out_dir, f"{base_name}.tex")
    with open(tex_path, "w") as f:
        f.write(tex_content)

    # 4. Rendered Vector PDF
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    render_table_to_pdf(df, title, pdf_path, bold_dict)

    print(f"[Exported] {base_name} -> .csv, .md, .tex, .pdf in {out_dir}")


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(results_dir, "analysis_tables"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    print("=" * 80)
    print("Exact Policy Evaluation: Publication Analysis & Reviewer Tables")
    print(f"Results Directory:  {results_dir}")
    print(f"Output Directory:   {out_dir}")
    print("=" * 80)

    baselines = load_baselines(project_root)
    df = load_dataset(results_dir, baselines)

    # Core Metrics for Reviewer Evaluation
    core_metrics = [
        ("auc_ve_mean", "Value Error (AUC)", True, "sci"),
        ("auc_cossim_mean", "Advantage CosSim", False, "decimal"),
        ("norm_vstart_auc", "Normalized V_start Return Score", False, "decimal"),
    ]

    win_rate_metrics = [
        ("auc_ve_mean", "auc_ve_sem", "Value Error (AUC)", True, "sci"),
        ("final_ve_mean", "final_ve_sem", "Value Error (Final)", True, "sci"),
        ("auc_cossim_mean", "auc_cossim_sem", "Advantage CosSim", False, "decimal"),
        ("norm_vstart_auc", "norm_vstart_auc_sem", "Normalized Greedy Return", False, "decimal"),
    ]

    key_matchups = [
        ("e0", "td0", "E(0) vs. TD(0) [Fair λ=0 peers]"),
        ("e_lambda", "td_lambda", "E(λ) vs. TD(λ) [Fair λ=0.8 peers]"),
        ("e0", "mc", "E(0) vs. MC [Dirichlet 1-step vs. Supervised]"),
        ("e_lambda", "mc", "E(λ) vs. MC [Dirichlet multi-step vs. Supervised]"),
        ("td0", "mc", "TD(0) vs. MC [Standard TD vs. Supervised]"),
    ]

    core_triad = ["td0", "mc", "e0"]
    all_5_algos = ["mc", "td0", "e0", "td_lambda", "e_lambda"]

    for k_val in [8, 64]:
        cap_label = "Bottleneck Capacity (k=8)" if k_val == 8 else "High Capacity (k=64)"

        # Table 1: Core Triad [TD(0) vs MC vs E(0)]
        df_triad, bold_triad = generate_comparison_table(
            df,
            algo_list=core_triad,
            k_val=k_val,
            metrics=core_metrics,
        )
        save_table_bundle(
            df=df_triad,
            bold_dict=bold_triad,
            base_name=f"table1_core_triad_k{k_val}",
            title=f"Core Triad Comparison: TD(0) vs. MC vs. E(0) [{cap_label}]",
            caption=f"Head-to-head comparison between TD(0), supervised Monte Carlo (MC), and Dirichlet-regularized value learning E(0) under {cap_label}. Wins and average ranks show strict dominance per environment. Best numbers in bold.",
            out_dir=out_dir,
        )

        # Table 2: All 5 Algorithms
        df_all, bold_all = generate_comparison_table(
            df,
            algo_list=all_5_algos,
            k_val=k_val,
            metrics=core_metrics,
        )
        save_table_bundle(
            df=df_all,
            bold_dict=bold_all,
            base_name=f"table2_all_5_algos_k{k_val}",
            title=f"Full Spectrum Comparison: MC, TD(0), E(0), TD(λ=0.8), E(λ=0.8) [{cap_label}]",
            caption=f"Comprehensive evaluation across all five algorithms under {cap_label}. Metrics include Value Error AUC (lower is better), Advantage Cosine Similarity (higher is better), and Normalized Greedy Policy Return (0 = Random, 1 = Optimal). Best numbers in bold.",
            out_dir=out_dir,
        )

        # Table 3: Pairwise Tournament Matrix (Reviewer Diagnostic)
        df_tourn, bold_tourn = generate_pairwise_tournament_matrix(df, k_val=k_val)
        save_table_bundle(
            df=df_tourn,
            bold_dict=bold_tourn,
            base_name=f"table3_pairwise_tournament_k{k_val}",
            title=f"Pairwise Head-to-Head Tournament Matrix for Value Error [{cap_label}]",
            caption=f"Pairwise head-to-head win-loss record (row vs. column) across all 9 environments on Value Error AUC under {cap_label}. Bold denotes a winning pairwise record.",
            out_dir=out_dir,
        )

        # Table 4: Core Triad Win Rate by Metric (Accounting for Statistical Ties)
        df_wr_triad, bold_wr_triad = generate_env_win_rate_table(
            df,
            algo_list=core_triad,
            k_val=k_val,
            metrics=win_rate_metrics,
        )
        save_table_bundle(
            df=df_wr_triad,
            bold_dict=bold_wr_triad,
            base_name=f"table4_env_win_rate_triad_k{k_val}",
            title=f"Core Triad Win Rate by Metric with Statistical Ties [{cap_label}]",
            caption=f"Effective win rate (%) and breakdown [outright wins + shared ties] for each algorithm in the core triad (TD(0), MC, E(0)) across each evaluation metric under {cap_label}. Statistical ties are defined by 1-SEM overlap with the top performer and split equally. Bold indicates top algorithm.",
            out_dir=out_dir,
        )

        # Table 5: All 5 Algorithms Win Rate by Metric (Accounting for Statistical Ties)
        df_wr_all, bold_wr_all = generate_env_win_rate_table(
            df,
            algo_list=all_5_algos,
            k_val=k_val,
            metrics=win_rate_metrics,
        )
        save_table_bundle(
            df=df_wr_all,
            bold_dict=bold_wr_all,
            base_name=f"table5_env_win_rate_all_k{k_val}",
            title=f"Full Spectrum Win Rate by Metric with Statistical Ties [{cap_label}]",
            caption=f"Effective win rate (%) and breakdown [outright wins + shared ties] across all five algorithms under {cap_label}. Statistical ties are defined by 1-SEM overlap with the top performer and split equally. Bold indicates top algorithm.",
            out_dir=out_dir,
        )

        # Table 6: Pairwise Head-to-Head with Statistical Ties (Wins - Losses - Ties)
        df_h2h, bold_h2h = generate_pairwise_head_to_head_table(
            df,
            matchups=key_matchups,
            k_val=k_val,
            metrics=win_rate_metrics,
        )
        save_table_bundle(
            df=df_h2h,
            bold_dict=bold_h2h,
            base_name=f"table6_head_to_head_statistical_ties_k{k_val}",
            title=f"Pairwise Head-to-Head Records with Statistical Ties [Wins - Losses - Ties] [{cap_label}]",
            caption=f"Head-to-head records across all 9 environments under {cap_label}. Each cell displays Wins - Losses - Ties (Effective Win %) where a tie denotes overlapping 1-SEM intervals (|val_A - val_B| <= SEM_A + SEM_B). Bold indicates a winning record (W > L).",
            out_dir=out_dir,
        )

    print("\n" + "=" * 80)
    print(f"[Done] All publication tables successfully generated in: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
