"""
scripts/sampled_E_surgical_ppo_sweep/analyze_results.py

Comprehensive publication-focused data analysis for Sampled E-Surgical PPO Sweeps.
Generates:
1. Benchmark comparison tables for:
   - Standardized Policy Return (AUC and Final)
   - Raw Reward (AUC and Final)
   - Start-State Value (V_start AUC and Final, if logged)
2. Pairwise Head-to-Head Win-Loss-Tie matrices.
3. Multi-format export:
   - Markdown (.md)
   - LaTeX booktabs (.tex)
   - PDF (.pdf) via pdflatex / matplotlib rendering.
All saved into <results_dir>/analysis_tables/
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import subprocess
import shutil
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

ALGO_ORDER = ["mc", "td_lambda", "oct8_biased", "forward_e", "surgical_v_aux"]

ALGO_DISPLAY_NAMES = {
    "mc": "MC (Sampled)",
    "td_lambda": "TD(λ=0.9)",
    "oct8_biased": "OG Oct 8 (Biased)",
    "forward_e": "Forward-Only E",
    "surgical_v_aux": "Surgical V_aux",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Sampled E-Surgical PPO Results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/sampled_E_surgical_ppo_sweep",
        help="Root results directory containing environment subfolders or summary CSVs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for tables (default: <results_dir>/analysis_tables)",
    )
    return parser.parse_args()


def load_data(results_dir: str) -> pd.DataFrame:
    summary_dfs = []
    csv_master = os.path.join(results_dir, "summary_table_all.csv")
    if os.path.exists(csv_master):
        try:
            return pd.read_csv(csv_master)
        except Exception:
            pass

    for env_name in ALL_ENVS:
        sum_path = os.path.join(results_dir, env_name, "summary.csv")
        if os.path.exists(sum_path):
            try:
                df = pd.read_csv(sum_path)
                summary_dfs.append(df)
            except Exception:
                pass

    if summary_dfs:
        master_df = pd.concat(summary_dfs, ignore_index=True)
        return master_df
    return pd.DataFrame()


def render_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    n_cols = len(df.columns)
    col_spec = "l" + "c" * (n_cols - 1)
    header = " & ".join([f"\\textbf{{{c}}}" for c in df.columns]) + " \\\\"

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
    ]

    for _, row in df.iterrows():
        row_str = " & ".join([str(val).replace("_", "\\_") for val in row]) + " \\\\"
        lines.append(row_str)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def render_table_to_pdf_matplotlib(df: pd.DataFrame, pdf_path: str, title: str):
    n_rows, n_cols = df.shape
    fig_width = max(10, n_cols * 2.2)
    fig_height = max(4, (n_rows + 2) * 0.45)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.axis("tight")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#2b5c8f")
        cell.set_edgecolor("#ffffff")

    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor("#f2f5f9" if i % 2 == 0 else "#ffffff")
            cell.set_edgecolor("#e0e0e0")

    plt.title(title, fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def compile_latex_to_pdf(latex_code: str, pdf_path: str, df: pd.DataFrame, title: str):
    pdflatex_bin = shutil.which("pdflatex")
    if pdflatex_bin is None and os.path.exists("/Library/TeX/texbin/pdflatex"):
        pdflatex_bin = "/Library/TeX/texbin/pdflatex"

    success = False
    if pdflatex_bin:
        tex_dir = os.path.dirname(pdf_path)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        standalone_tex = os.path.join(tex_dir, f"{base_name}_standalone.tex")

        doc = [
            "\\documentclass[preview,border=12pt]{standalone}",
            "\\usepackage{booktabs}",
            "\\usepackage{amsmath}",
            "\\usepackage{array}",
            "\\begin{document}",
            latex_code,
            "\\end{document}",
        ]
        try:
            with open(standalone_tex, "w") as f:
                f.write("\n".join(doc))

            res = subprocess.run(
                [pdflatex_bin, "-interaction=nonstopmode", f"{base_name}_standalone.tex"],
                cwd=tex_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            compiled_pdf = os.path.join(tex_dir, f"{base_name}_standalone.pdf")
            if os.path.exists(compiled_pdf):
                shutil.move(compiled_pdf, pdf_path)
                success = True
            for ext in [".aux", ".log", "_standalone.tex"]:
                f_del = os.path.join(tex_dir, f"{base_name}{ext}")
                if os.path.exists(f_del):
                    os.remove(f_del)
        except Exception:
            success = False

    if not success:
        render_table_to_pdf_matplotlib(df, pdf_path, title)


def export_multi_format(df: pd.DataFrame, base_path: str, caption: str, label: str):
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

    compile_latex_to_pdf(latex_str, f"{base_path}.pdf", df, caption)
    print(f"[Export] Saved table: {base_path} (.csv, .md, .tex, .pdf)")


def build_benchmark_table(
    df: pd.DataFrame,
    mean_col: str,
    sem_col: str,
    higher_is_better: bool = True,
    algos: list[str] = ALGO_ORDER,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    present_envs = [e for e in ALL_ENVS if e in df["env_name"].unique()]
    rows = []

    for env_name in present_envs:
        env_rows = df[df["env_name"] == env_name]
        if env_rows.empty:
            continue

        row = {"Environment": ENV_DISPLAY_NAMES.get(env_name, env_name)}
        metric_dict = {}

        for algo in algos:
            algo_row = env_rows[env_rows["algo"] == algo]
            display = ALGO_DISPLAY_NAMES.get(algo, algo)
            if not algo_row.empty:
                mean = algo_row[mean_col].values[0]
                sem = algo_row[sem_col].values[0] if sem_col in algo_row and not np.isnan(algo_row[sem_col].values[0]) else 0.0
                row[display] = f"{mean:.4f} ± {sem:.4f}"
                metric_dict[algo] = mean
            else:
                row[display] = "N/A"

        if metric_dict:
            best_algo = max(metric_dict, key=metric_dict.get) if higher_is_better else min(metric_dict, key=metric_dict.get)
            row["Winner"] = ALGO_DISPLAY_NAMES.get(best_algo, best_algo)
        else:
            row["Winner"] = "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


def compute_win_loss_rates(df: pd.DataFrame, mean_col: str, sem_col: str, higher_is_better: bool = True) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    algos = [a for a in ALGO_ORDER if a in df["algo"].unique()]
    n = len(algos)
    wlt_matrix = np.zeros((n, n, 3), dtype=int)

    present_envs = [e for e in ALL_ENVS if e in df["env_name"].unique()]

    for env_name in present_envs:
        env_rows = df[df["env_name"] == env_name]
        for i, a1 in enumerate(algos):
            r1 = env_rows[env_rows["algo"] == a1]
            if r1.empty:
                continue
            m1 = r1[mean_col].values[0]
            s1 = r1[sem_col].values[0] if sem_col in r1 and not np.isnan(r1[sem_col].values[0]) else 0.0

            for j, a2 in enumerate(algos):
                if i == j:
                    continue
                r2 = env_rows[env_rows["algo"] == a2]
                if r2.empty:
                    continue
                m2 = r2[mean_col].values[0]
                s2 = r2[sem_col].values[0] if sem_col in r2 and not np.isnan(r2[sem_col].values[0]) else 0.0

                pooled_sem = np.sqrt(s1**2 + s2**2)
                diff = m1 - m2

                if abs(diff) <= pooled_sem:
                    wlt_matrix[i, j, 2] += 1
                elif (diff > 0 if higher_is_better else diff < 0):
                    wlt_matrix[i, j, 0] += 1
                else:
                    wlt_matrix[i, j, 1] += 1

    rows = []
    for i, a1 in enumerate(algos):
        row = {"Algorithm": ALGO_DISPLAY_NAMES.get(a1, a1)}
        for j, a2 in enumerate(algos):
            if i == j:
                row[ALGO_DISPLAY_NAMES.get(a2, a2)] = "-"
            else:
                w, l, t = wlt_matrix[i, j]
                row[ALGO_DISPLAY_NAMES.get(a2, a2)] = f"{w}-{l}-{t}"
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir if args.out_dir else os.path.join(results_dir, "analysis_tables"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    df = load_data(results_dir)

    if df.empty:
        print("[Error] No results found to analyze. Check results directory.")
        return

    # 1. Standardized Performance Benchmark Table (AUC, higher is better)
    t_perf_auc = build_benchmark_table(
        df, mean_col="std_perf_auc_mean", sem_col="std_perf_auc_sem", higher_is_better=True
    )
    if not t_perf_auc.empty:
        export_multi_format(
            t_perf_auc,
            os.path.join(out_dir, "table_std_perf_auc_benchmark"),
            "Sampled E-Surgical PPO Standardized Return AUC Benchmark",
            "tab:std_perf_auc_benchmark",
        )

    # 2. Final Standardized Performance Benchmark Table (higher is better)
    t_perf_final = build_benchmark_table(
        df, mean_col="std_perf_final_mean", sem_col="std_perf_final_sem", higher_is_better=True
    )
    if not t_perf_final.empty:
        export_multi_format(
            t_perf_final,
            os.path.join(out_dir, "table_std_perf_final_benchmark"),
            "Sampled E-Surgical PPO Final Standardized Return Benchmark",
            "tab:std_perf_final_benchmark",
        )

    # 3. Start-State Value Benchmark Table (V_start AUC, if present)
    if "vstart_auc_mean" in df.columns:
        t_vstart = build_benchmark_table(
            df, mean_col="vstart_auc_mean", sem_col="vstart_auc_sem", higher_is_better=True
        )
        if not t_vstart.empty:
            export_multi_format(
                t_vstart,
                os.path.join(out_dir, "table_vstart_auc_benchmark"),
                "Sampled E-Surgical PPO Start-State Value Benchmark (AUC)",
                "tab:vstart_auc_benchmark",
            )

    # 4. Pairwise Win-Loss-Tie Matrices on Performance AUC
    wlt_perf = compute_win_loss_rates(
        df, mean_col="std_perf_auc_mean", sem_col="std_perf_auc_sem", higher_is_better=True
    )
    if not wlt_perf.empty:
        export_multi_format(
            wlt_perf,
            os.path.join(out_dir, "table_pairwise_wlt_std_perf"),
            "Pairwise Head-to-Head Win-Loss-Tie Matrix on Standardized Return AUC",
            "tab:pairwise_wlt_std_perf",
        )

    print(f"\nAll publication tables generated in: {out_dir}")


if __name__ == "__main__":
    main()
