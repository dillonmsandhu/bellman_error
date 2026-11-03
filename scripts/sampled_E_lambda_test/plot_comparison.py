"""
scripts/sampled_E_lambda_test/plot_comparison.py

Consolidates and plots Sampling E(lambda) Test results across benchmark environments.
Compares the 3 formulations against Exact Matrix E(lambda):
1. Naive Sampling (Oct 8 Analog: differentiates both ends of jump)
2. Aux Version (v_aux debiasing)
3. Forward Only (drops backward branch)

Figures generated:
1. aggregate_gradient_metrics_across_envs.png: Grouped bar chart comparing Cosine Alignment, MSE, and Variance
2. gradient_mse_across_envs.png: Time series of Gradient MSE to Exact E(lambda) across updates
3. gradient_cosine_across_envs.png: Time series of Cosine Similarity to Exact E(lambda) across updates
4. gradient_variance_across_envs.png: Time series of Gradient Variance across updates
5. loss_E_variants_across_envs.png: Exact E(lambda) vs Naive vs Aux vs Forward Only values
6. summary_metrics.csv: Full statistical summary table across environments
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import cloudpickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"

ALL_ENVS = [
    "fourrooms-dense",
    "FourRooms-misc",
    "fourrooms-mines",
    "fourrooms-mines-dense",
    "eightrooms-dense",
    "eightrooms-misc",
    "whirlpool-misc",
    "SpaceInvadersExactValue",
    "mountaincar-dense",
]

VAR_COLORS = {
    "naive": "#2ca02c",     # Green (Oct 8 analog)
    "aux": "#9467bd",       # Purple (Drop Invalid Jump)
    "aux_mc": "#d62728",    # Red / Coral (MC Invalid Jump)
    "aux_full_mc": "#ff7f0e", # Orange (Full MC Invalid Jump)
    "fwd": "#00bcd4",       # Cyan/Teal (forward only)
    "exact": "#000000",     # Black
}

VAR_LABELS = {
    "naive": "Naive (Oct 8 Analog)",
    "aux": "V_aux (Drop Invalid Jump)",
    "aux_mc": "V_aux (MC Invalid Jump)",
    "aux_full_mc": "V_aux (Full MC Invalid)",
    "fwd": "Forward Only",
}

VAR_LABELS_TEX = {
    "naive": "Naive (Oct 8 Analog)",
    "aux": r"$V_{\mathrm{aux}}$ (Drop Invalid Jump)",
    "aux_mc": r"$V_{\mathrm{aux}}$ (MC Invalid Jump)",
    "aux_full_mc": r"$V_{\mathrm{aux}}$ (Full MC Invalid)",
    "fwd": "Forward Only",
}



def load_env_metrics(results_dir: str, env_name: str) -> List[Dict[str, np.ndarray]]:
    pattern = os.path.join(results_dir, env_name, "*.pkl")
    files = glob.glob(pattern)
    metrics_list = []
    for fp in sorted(files):
        try:
            with open(fp, "rb") as f:
                data = cloudpickle.load(f)
                if isinstance(data, dict) and "metrics" in data:
                    metrics_list.append(data["metrics"])
                elif isinstance(data, dict):
                    metrics_list.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {fp}: {e}")
    return metrics_list


def aggregate_seeds(metrics_list: List[Dict[str, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    agg = {}
    for k in keys:
        try:
            raw_arrays = [np.asarray(m[k], dtype=np.float64) for m in metrics_list if k in m]
            if not raw_arrays:
                continue

            expanded = []
            for a in raw_arrays:
                if a.ndim == 1:
                    expanded.append(a[None, :])
                elif a.ndim >= 2:
                    expanded.append(a.reshape(-1, a.shape[-1]))

            min_len = min(a.shape[1] for a in expanded)
            all_seeds = np.concatenate([a[:, :min_len] for a in expanded], axis=0)

            mean = np.nanmean(all_seeds, axis=0)
            sem = np.nanstd(all_seeds, axis=0) / np.sqrt(max(1, all_seeds.shape[0]))
            agg[k] = (mean, sem)
        except Exception:
            pass
    return agg


def plot_aggregate_bar_chart(all_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], out_dir: str):
    envs = [e for e in ALL_ENVS if e in all_data]
    if not envs:
        return

    n_envs = len(envs)
    active_vars = [
        v for v in VAR_LABELS.keys()
        if any(f"rho_samp_{v}_exact" in all_data[e] or f"var_samp_{v}" in all_data[e] or f"mse_samp_{v}" in all_data[e] for e in envs)
    ]
    n_vars = max(1, len(active_vars))
    width = min(0.18, 0.8 / n_vars)
    center_offset = (n_vars - 1) / 2.0 * width

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    x = np.arange(n_envs)

    # 1. Cosine Similarity to Exact E(lambda)
    ax = axes[0]
    for idx, var in enumerate(active_vars):
        label = VAR_LABELS[var]
        key = f"rho_samp_{var}_exact"
        means = [np.mean(all_data[e][key][0]) if key in all_data[e] else np.nan for e in envs]
        errs = [np.mean(all_data[e][key][1]) if key in all_data[e] else 0.0 for e in envs]
        ax.bar(x + idx * width, means, width, yerr=errs, label=label, color=VAR_COLORS[var], capsize=3, edgecolor="black")

    ax.set_title(r"Cosine Similarity to Exact $\nabla E(\lambda)$", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$\cos(\bar{g}, g^*)$", fontsize=11)
    ax.set_xticks(x + center_offset)
    ax.set_xticklabels(envs, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=True, fontsize=8)

    # 2. Gradient Variance
    ax = axes[1]
    for idx, var in enumerate(active_vars):
        label = VAR_LABELS[var]
        key = f"var_samp_{var}"
        means = [np.mean(all_data[e][key][0]) if key in all_data[e] else np.nan for e in envs]
        ax.bar(x + idx * width, means, width, label=label, color=VAR_COLORS[var], edgecolor="black")

    ax.set_title(r"Gradient Estimator Variance $\mathrm{Var}(\bar{g})$", fontsize=12, fontweight="bold")
    ax.set_ylabel("Variance (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_xticks(x + center_offset)
    ax.set_xticklabels(envs, rotation=35, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Gradient MSE
    ax = axes[2]
    for idx, var in enumerate(active_vars):
        label = VAR_LABELS[var]
        key = f"mse_samp_{var}"
        means = [np.mean(all_data[e][key][0]) if key in all_data[e] else np.nan for e in envs]
        ax.bar(x + idx * width, means, width, label=label, color=VAR_COLORS[var], edgecolor="black")

    ax.set_title(r"Gradient MSE $||\bar{g} - g^*||^2$", fontsize=12, fontweight="bold")
    ax.set_ylabel("MSE (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_xticks(x + center_offset)
    ax.set_xticklabels(envs, rotation=35, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Sampled E(lambda) Formulations: Exact Alignment & Variance across Benchmark Environments", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "aggregate_gradient_metrics_across_envs.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


def plot_metric_time_series(
    all_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    metric_prefix: str,
    title: str,
    ylabel: str,
    filename: str,
    out_dir: str,
    log_scale: bool = False,
):
    envs = [e for e in ALL_ENVS if e in all_data]
    if not envs:
        return

    n_envs = len(envs)
    cols = min(3, n_envs)
    rows = int(np.ceil(n_envs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for i, env in enumerate(envs):
        ax = axes[i]
        for var, label in VAR_LABELS.items():
            key = f"{metric_prefix}_{var}" if f"{metric_prefix}_{var}" in all_data[env] else f"{metric_prefix}_{var}_exact"
            if key in all_data[env]:
                mean, std = all_data[env][key]
                mean = np.squeeze(np.asarray(mean))
                std = np.squeeze(np.asarray(std))
                if mean.ndim == 0:
                    mean = mean[None]
                    std = std[None]
                steps = np.arange(len(mean))
                ax.plot(steps, mean, label=label, color=VAR_COLORS[var], lw=1.8)
                if std is not None and np.any(std > 0):
                    ax.fill_between(steps, mean - std, mean + std, color=VAR_COLORS[var], alpha=0.15)

        ax.set_title(env, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        if "rho" in metric_prefix or "cos" in metric_prefix:
            ax.axhline(1.0, color="black", linestyle=":", lw=1.0, alpha=0.6)
            ax.set_ylim(-0.3, 1.05)
        if log_scale:
            ax.set_yscale("log")
        if i == 0 and len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(frameon=True, fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


def plot_individual_bar_charts(all_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], out_dir: str):
    """
    Creates individual, publication-ready bar charts for:
    1. Cosine Similarity to Exact Matrix E(lambda)
    2. Gradient Variance Var(g)
    3. Gradient MSE ||g - g*||^2
    4. Aggregated 3-metric summary across all environments
    Saves both PNG and PDF for each figure.
    """
    envs = [e for e in ALL_ENVS if e in all_data]
    if not envs:
        return

    n_envs = len(envs)
    variants = [
        v for v in VAR_LABELS.keys()
        if any(f"rho_samp_{v}_exact" in all_data[e] or f"var_samp_{v}" in all_data[e] or f"mse_samp_{v}" in all_data[e] for e in envs)
    ]
    n_vars = max(1, len(variants))
    width = min(0.18, 0.75 / n_vars)

    # Precompute per-env and aggregated values
    # Metric specs: (metric_base, title, ylabel, fname, log_scale, y_limits, hlines)
    metric_specs = [
        (
            "rho_samp",
            r"Cosine Similarity to Exact $\nabla E(\lambda)$ across Benchmark Environments",
            r"Cosine Similarity $\cos(\bar{g}, g^*)$",
            "bar_cosine_similarity",
            False,
            (-0.35, 1.05),
            [0.0, 1.0],
        ),
        (
            "var_samp",
            r"Gradient Estimator Variance $\mathrm{Var}(\bar{g})$ across Benchmark Environments",
            r"Variance $\mathrm{Var}(\bar{g})$ (log scale)",
            "bar_gradient_variance",
            True,
            None,
            [],
        ),
        (
            "mse_samp",
            r"Gradient MSE $||\bar{g} - g^*||^2$ to Exact $\nabla E(\lambda)$ across Environments",
            r"Gradient MSE $||\bar{g} - g^*||^2$ (log scale)",
            "bar_gradient_mse",
            True,
            None,
            [],
        ),
    ]

    # Indices: 0 .. n_envs-1 for envs, then a gap, then n_envs for Aggregated Mean
    x_positions = np.arange(n_envs + 1)
    # Give the aggregated group a slightly wider gap
    x_coords = np.zeros(n_envs + 1)
    x_coords[:n_envs] = np.arange(n_envs)
    x_coords[n_envs] = n_envs + 0.35
    xtick_labels = envs + ["Mean (All Envs)"]

    for prefix, title, ylabel, fname, log_scale, ylim, hlines in metric_specs:
        fig, ax = plt.subplots(figsize=(13, 5.5))

        for v_idx, var in enumerate(variants):
            key = f"{prefix}_{var}_exact" if f"{prefix}_{var}_exact" in all_data[envs[0]] else f"{prefix}_{var}"
            vals = []
            errs = []
            for e in envs:
                if key in all_data[e]:
                    m_arr = all_data[e][key][0]
                    s_arr = all_data[e][key][1]
                    vals.append(float(np.nanmean(m_arr)))
                    errs.append(float(np.nanmean(s_arr)))
                else:
                    vals.append(np.nan)
                    errs.append(0.0)

            # Compute across-env mean and standard error of the mean
            agg_mean = float(np.nanmean(vals))
            agg_sem = float(np.nanstd(vals) / np.sqrt(max(1, len(vals))))
            plot_vals = vals + [agg_mean]
            plot_errs = errs + [agg_sem]

            offsets = x_coords + (v_idx - (n_vars - 1) / 2.0) * width
            bars = ax.bar(
                offsets,
                plot_vals,
                width,
                yerr=plot_errs,
                label=VAR_LABELS[var],
                color=VAR_COLORS[var],
                capsize=3,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.92,
            )

            # Emphasize the aggregated bar border
            bars[-1].set_linewidth(1.6)

        # Highlight the Aggregated group background
        ax.axvspan(x_coords[-1] - 0.45, x_coords[-1] + 0.45, color="gray", alpha=0.10, zorder=0)
        ax.axvline(x_coords[-1] - 0.55, color="black", linestyle="--", linewidth=1.0, alpha=0.5)

        for hl in hlines:
            ax.axhline(hl, color="black", linestyle=":" if hl > 0 else "-", linewidth=1.0, alpha=0.6)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="medium")
        ax.set_xticks(x_coords)
        ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=9.5)

        if log_scale:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(frameon=True, fontsize=9, loc="best", ncol=2)
        plt.tight_layout()

        # Save both PNG and PDF
        png_path = os.path.join(out_dir, f"{fname}.png")
        pdf_path = os.path.join(out_dir, f"{fname}.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {png_path} and {pdf_path}")

    # 4. Aggregated 3-metric summary figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    metrics_summary = [
        ("rho_samp", r"Cosine Alignment $\cos(\bar{g}, g^*)$", False, (-0.1, 1.05)),
        ("var_samp", r"Gradient Variance $\mathrm{Var}(\bar{g})$", True, None),
        ("mse_samp", r"Gradient MSE $||\bar{g} - g^*||^2$", True, None),
    ]

    for ax_idx, (prefix, ax_title, log_scale, ylim) in enumerate(metrics_summary):
        ax = axes[ax_idx]
        agg_vals = []
        agg_errs = []
        for var in variants:
            key = f"{prefix}_{var}_exact" if f"{prefix}_{var}_exact" in all_data[envs[0]] else f"{prefix}_{var}"
            vals = [float(np.nanmean(all_data[e][key][0])) for e in envs if key in all_data[e]]
            agg_vals.append(float(np.nanmean(vals)))
            agg_errs.append(float(np.nanstd(vals) / np.sqrt(max(1, len(vals)))))

        bars = ax.bar(
            np.arange(n_vars),
            agg_vals,
            width=0.55,
            yerr=agg_errs,
            color=[VAR_COLORS[v] for v in variants],
            edgecolor="black",
            linewidth=1.0,
            capsize=4,
            alpha=0.92,
        )
        ax.set_title(ax_title, fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(n_vars))
        ax.set_xticklabels([VAR_LABELS[v] for v in variants], rotation=25, ha="right", fontsize=8.5)
        if log_scale:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(r"Sampled $E(\lambda)$: Aggregated Metric Summary Across Benchmark Environments", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    summary_png = os.path.join(out_dir, "bar_aggregated_summary.png")
    summary_pdf = os.path.join(out_dir, "bar_aggregated_summary.pdf")
    plt.savefig(summary_png, dpi=300, bbox_inches="tight")
    plt.savefig(summary_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {summary_png} and {summary_pdf}")


def export_latex_plots_and_tables(all_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], out_dir: str):
    """
    Generates standalone compilable LaTeX TikZ/PGFPlots (.tex) and LaTeX booktabs tables (.tex),
    and compiles them directly with pdflatex into vector PDF documents.
    """
    import subprocess
    import shutil

    latex_dir = os.path.join(out_dir, "latex_plots")
    os.makedirs(latex_dir, exist_ok=True)

    envs = [e for e in ALL_ENVS if e in all_data]
    if not envs:
        return

    n_envs = len(envs)
    variants = [
        v for v in VAR_LABELS.keys()
        if any(f"rho_samp_{v}_exact" in all_data[e] or f"var_samp_{v}" in all_data[e] or f"mse_samp_{v}" in all_data[e] for e in envs)
    ]
    pdflatex_bin = "/Library/TeX/texbin/pdflatex"
    has_pdflatex = os.path.exists(pdflatex_bin) or (shutil.which("pdflatex") is not None)
    if not os.path.exists(pdflatex_bin) and shutil.which("pdflatex"):
        pdflatex_bin = shutil.which("pdflatex")

    # Safe LaTeX short names for environments
    env_tex_names = [e.replace("_", r"\_") for e in envs] + ["Aggregated"]

    # Compute data matrix for all metrics and variants
    # structure: metric_data[metric][var] = list of float values (length n_envs + 1)
    metric_prefixes = ["rho_samp", "var_samp", "mse_samp"]
    data_table: Dict[str, Dict[str, List[float]]] = {m: {v: [] for v in variants} for m in metric_prefixes}

    for prefix in metric_prefixes:
        for var in variants:
            key = f"{prefix}_{var}_exact" if f"{prefix}_{var}_exact" in all_data[envs[0]] else f"{prefix}_{var}"
            env_vals = []
            for e in envs:
                if key in all_data[e]:
                    env_vals.append(float(np.nanmean(all_data[e][key][0])))
                else:
                    env_vals.append(np.nan)
            agg_val = float(np.nanmean(env_vals))
            data_table[prefix][var] = env_vals + [agg_val]

    # 1. PGFPlots Grouped Bar Charts for Per-Env Metrics
    tikz_specs = [
        (
            "rho_samp",
            "plot_cosine_per_env",
            "Cosine Similarity $\\cos(\\bar{g}, g^*)$",
            False,
            "ymin=-0.35, ymax=1.05",
            [0.0, 1.0],
        ),
        (
            "var_samp",
            "plot_variance_per_env",
            "Variance $\\mathrm{Var}(\\bar{g})$ (log scale)",
            True,
            "ymin=1e-6, ymax=1e0",
            [],
        ),
        (
            "mse_samp",
            "plot_mse_per_env",
            "Gradient MSE $||\\bar{g} - g^*||^2$ (log scale)",
            True,
            "ymin=1e-6, ymax=2e1",
            [],
        ),
    ]

    for prefix, base_name, ylabel, is_log, yrange, hlines in tikz_specs:
        tex_path = os.path.join(latex_dir, f"{base_name}.tex")
        pdf_path = os.path.join(latex_dir, f"{base_name}.pdf")

        # Format xtick labels for TikZ
        xtick_indices = ",".join(str(i) for i in range(len(env_tex_names)))
        xtick_labels_tex = "{" + ",".join(f"{{{name}}}" for name in env_tex_names) + "}"

        extra_lines = ""
        for hl in hlines:
            extra_lines += f"\\addplot[black, densely dotted, line width=0.8pt, domain=-0.5:{len(env_tex_names)-0.5}] {{{hl}}};\n"

        plots_code = []
        tex_colors = {
            "naive": "cnaive",
            "aux": "caux",
            "aux_mc": "cauxmc",
            "aux_full_mc": "cauxfullmc",
            "fwd": "cfwd",
        }
        for var in variants:
            cname = tex_colors[var]
            coords = []
            for i, val in enumerate(data_table[prefix][var]):
                safe_val = max(1e-7, val) if is_log else val
                coords.append(f"({i}, {safe_val:.6g})")
            coords_str = " ".join(coords)
            plots_code.append(
                f"\\addplot[fill={cname}, draw=black!80, line width=0.4pt] coordinates {{{coords_str}}};\n"
                f"\\addlegendentry{{{VAR_LABELS_TEX[var]}}}"
            )

        plots_block = "\n".join(plots_code)

        axis_opts_list = [
            "ybar",
            "bar width=4.5pt",
            "width=16cm",
            "height=7.5cm",
            f"ylabel={{{ylabel}}}",
            f"xtick={{{xtick_indices}}}",
            f"xticklabels={xtick_labels_tex}",
            r"x tick label style={rotate=28, anchor=east, font=\footnotesize}",
        ]
        if is_log:
            axis_opts_list.append("ymode=log, log basis y=10")
        if yrange:
            axis_opts_list.append(yrange)
        axis_opts_list.extend([
            "legend pos=north west",
            r"legend style={font=\scriptsize, fill=white, fill opacity=0.88, draw=black!40, legend columns=2}",
            "ymajorgrids=true",
            r"grid style={dashed, gray!30}",
            "enlarge x limits=0.08",
        ])
        axis_opts_str = ",\n    ".join(axis_opts_list)

        tex_content = f"""\\documentclass[tikz,border=6pt]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\definecolor{{cnaive}}{{HTML}}{{2CA02C}}
\\definecolor{{caux}}{{HTML}}{{9467BD}}
\\definecolor{{cauxmc}}{{HTML}}{{D62728}}
\\definecolor{{cauxfullmc}}{{HTML}}{{FF7F0E}}
\\definecolor{{cfwd}}{{HTML}}{{00BCD4}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
    {axis_opts_str}
]
{extra_lines}
{plots_block}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""
        with open(tex_path, "w") as f:
            f.write(tex_content)
        print(f"Saved: {tex_path}")

        if has_pdflatex:
            try:
                subprocess.run(
                    [pdflatex_bin, "-interaction=nonstopmode", f"{base_name}.tex"],
                    cwd=latex_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if os.path.exists(pdf_path):
                    print(f"Compiled: {pdf_path}")
            except Exception as e:
                print(f"Warning: pdflatex failed for {tex_path}: {e}")

    # 2. PGFPlots Aggregated Summary Bar Chart
    agg_tex_path = os.path.join(latex_dir, "plot_aggregated_summary.tex")
    agg_pdf_path = os.path.join(latex_dir, "plot_aggregated_summary.pdf")
    
    agg_cossim_coords = " ".join([f"({i}, {data_table['rho_samp'][v][-1]:.4f})" for i, v in enumerate(variants)])
    agg_var_coords = " ".join([f"({i}, {data_table['var_samp'][v][-1]:.6g})" for i, v in enumerate(variants)])
    agg_mse_coords = " ".join([f"({i}, {data_table['mse_samp'][v][-1]:.6g})" for i, v in enumerate(variants)])
    
    var_xticklabels = "{" + ",".join(f"{{{VAR_LABELS_TEX[v]}}}" for v in variants) + "}"
    agg_xtick_indices = ",".join(str(i) for i in range(len(variants)))

    agg_tex_content = f"""\\documentclass[tikz,border=6pt]{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\definecolor{{cnaive}}{{HTML}}{{2CA02C}}
\\definecolor{{caux}}{{HTML}}{{9467BD}}
\\definecolor{{cauxmc}}{{HTML}}{{D62728}}
\\definecolor{{cauxfullmc}}{{HTML}}{{FF7F0E}}
\\definecolor{{cfwd}}{{HTML}}{{00BCD4}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
    name=ax1,
    ybar,
    bar width=14pt,
    width=6.5cm,
    height=6.0cm,
    title={{\\textbf{{Cosine Similarity $\\cos(\\bar{{g}}, g^*)$}}}},
    title style={{font=\\small}},
    xtick={{{agg_xtick_indices}}},
    xticklabels={var_xticklabels},
    x tick label style={{rotate=30, anchor=east, font=\\scriptsize}},
    ymin=0.0, ymax=0.75,
    ymajorgrids=true,
    grid style={{dashed, gray!30}},
    point meta=y,
]
\\addplot+[ybar, fill=caux!70, draw=black!80] coordinates {{{agg_cossim_coords}}};
\\end{{axis}}

\\begin{{axis}}[
    name=ax2,
    at={{($(ax1.east)+(1.2cm,0)$)}},
    anchor=west,
    ybar,
    ymode=log,
    bar width=14pt,
    width=6.5cm,
    height=6.0cm,
    title={{\\textbf{{Gradient Variance $\\mathrm{{Var}}(\\bar{{g}})$}}}},
    title style={{font=\\small}},
    xtick={{{agg_xtick_indices}}},
    xticklabels={var_xticklabels},
    x tick label style={{rotate=30, anchor=east, font=\\scriptsize}},
    ymajorgrids=true,
    grid style={{dashed, gray!30}},
]
\\addplot+[ybar, fill=cnaive!70, draw=black!80] coordinates {{{agg_var_coords}}};
\\end{{axis}}

\\begin{{axis}}[
    name=ax3,
    at={{($(ax2.east)+(1.2cm,0)$)}},
    anchor=west,
    ybar,
    ymode=log,
    bar width=14pt,
    width=6.5cm,
    height=6.0cm,
    title={{\\textbf{{Gradient MSE $||\\bar{{g}} - g^*||^2$}}}},
    title style={{font=\\small}},
    xtick={{{agg_xtick_indices}}},
    xticklabels={var_xticklabels},
    x tick label style={{rotate=30, anchor=east, font=\\scriptsize}},
    ymajorgrids=true,
    grid style={{dashed, gray!30}},
]
\\addplot+[ybar, fill=cauxmc!70, draw=black!80] coordinates {{{agg_mse_coords}}};
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""
    with open(agg_tex_path, "w") as f:
        f.write(agg_tex_content)
    print(f"Saved: {agg_tex_path}")

    if has_pdflatex:
        try:
            subprocess.run(
                [pdflatex_bin, "-interaction=nonstopmode", "plot_aggregated_summary.tex"],
                cwd=latex_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if os.path.exists(agg_pdf_path):
                print(f"Compiled: {agg_pdf_path}")
        except Exception as e:
            print(f"Warning: pdflatex failed for {agg_tex_path}: {e}")

    # Helper formatters for compact tables
    def fmt_cos(val: float) -> str:
        if np.isnan(val):
            return "-"
        return f"{val:.2f}"

    def fmt_num(val: float) -> str:
        if np.isnan(val):
            return "-"
        if val == 0:
            return "0"
        if 0.01 <= abs(val) < 100:
            return f"{val:.2f}"
        s = f"{val:.1e}"
        return s.replace("e-0", "e-").replace("e+0", "e+")

    def clean_env_label(name: str) -> str:
        if "SpaceInvaders" in name:
            return "SpaceInvaders"
        return name.replace("_", r"\_")

    # 3. LaTeX Booktabs Full Per-Env & Aggregated Table
    tbl_tex_path = os.path.join(latex_dir, "table_metrics_by_env.tex")
    tbl_pdf_path = os.path.join(latex_dir, "table_metrics_by_env.pdf")

    table_rows = []
    for i, env_name in enumerate(envs):
        esc_env = clean_env_label(env_name)
        r_str = f"{esc_env}"
        
        env_best_cos = max(data_table["rho_samp"][k][i] for k in variants)
        env_best_var = min(data_table["var_samp"][k][i] for k in variants)
        env_best_mse = min(data_table["mse_samp"][k][i] for k in variants)

        # Cosine (higher is better)
        for v in variants:
            val = data_table["rho_samp"][v][i]
            formatted = fmt_cos(val)
            is_best = (formatted == fmt_cos(env_best_cos)) or np.isclose(val, env_best_cos, atol=1e-3)
            r_str += f" & \\textbf{{{formatted}}}" if is_best else f" & {formatted}"
        # Variance (lower is better)
        for v in variants:
            val = data_table["var_samp"][v][i]
            formatted = fmt_num(val)
            is_best = (formatted == fmt_num(env_best_var)) or np.isclose(val, env_best_var, atol=1e-8)
            r_str += f" & \\textbf{{{formatted}}}" if is_best else f" & {formatted}"
        # MSE (lower is better)
        for v in variants:
            val = data_table["mse_samp"][v][i]
            formatted = fmt_num(val)
            is_best = (formatted == fmt_num(env_best_mse)) or np.isclose(val, env_best_mse, atol=1e-8)
            r_str += f" & \\textbf{{{formatted}}}" if is_best else f" & {formatted}"
        r_str += r" \\"
        table_rows.append(r_str)

    # Aggregated Mean row
    agg_str = r"\textbf{Mean (All Envs)}"
    best_cos = max(data_table["rho_samp"][k][-1] for k in variants)
    best_var = min(data_table["var_samp"][k][-1] for k in variants)
    best_mse = min(data_table["mse_samp"][k][-1] for k in variants)

    for v in variants:
        val = data_table["rho_samp"][v][-1]
        formatted = fmt_cos(val)
        is_best = (formatted == fmt_cos(best_cos)) or np.isclose(val, best_cos, atol=1e-3)
        agg_str += f" & \\textbf{{{formatted}}}" if is_best else f" & {formatted}"
    for v in variants:
        val = data_table["var_samp"][v][-1]
        formatted = fmt_num(val)
        is_best = (formatted == fmt_num(best_var)) or np.isclose(val, best_var, atol=1e-5)
        agg_str += f" & \\textbf{{{formatted}}}" if is_best else f" & {formatted}"
    for v in variants:
        val = data_table["mse_samp"][v][-1]
        formatted = fmt_num(val)
        is_best = (formatted == fmt_num(best_mse)) or np.isclose(val, best_mse, atol=1e-4)
        agg_str += f" & \\textbf{{{formatted}}}" if is_best else f" & {formatted}"
    agg_str += r" \\"

    rows_code = "\n".join(table_rows)

    nv = len(variants)
    var_short_names = {
        "naive": "Naive",
        "aux": "Drop",
        "aux_mc": "MC",
        "aux_full_mc": "Full MC",
        "fwd": "Fwd",
    }
    col_spec = "l " + ("c" * nv + " ") * 3
    cm1_start, cm1_end = 2, 1 + nv
    cm2_start, cm2_end = cm1_end + 1, cm1_end + nv
    cm3_start, cm3_end = cm2_end + 1, cm2_end + nv
    subheaders = " & ".join(var_short_names.get(v, v) for v in variants)
    header_row = f"\\textbf{{Environment}} & {subheaders} & {subheaders} & {subheaders} \\\\"
    cmidrule_row = f"\\cmidrule(lr){{{cm1_start}-{cm1_end}}} \\cmidrule(lr){{{cm2_start}-{cm2_end}}} \\cmidrule(lr){{{cm3_start}-{cm3_end}}}"
    multicol_row = f"& \\multicolumn{{{nv}}}{{c}}{{\\textbf{{Cosine Sim}} $\\uparrow$}} & \\multicolumn{{{nv}}}{{c}}{{\\textbf{{Variance}} $\\downarrow$}} & \\multicolumn{{{nv}}}{{c}}{{\\textbf{{MSE}} $\\downarrow$}} \\\\"

    table_tex_content = f"""\\documentclass[preview,border=6pt]{{standalone}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}

\\begin{{document}}
\\begin{{table}}[htbp]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\caption{{Empirical comparison of Sampled $E(\\lambda)$ gradient formulations against Exact Matrix $\\nabla E(\\lambda)$ across benchmark environments ($\\gamma=0.99$). Cosine similarity measures directional alignment ($\\uparrow$), while Variance and MSE measure estimator dispersion and accuracy ($\\downarrow$).}}
\\vspace{{4pt}}
\\begin{{tabular}}{{{col_spec.strip()}}}
\\toprule
{multicol_row}
{cmidrule_row}
{header_row}
\\midrule
{rows_code}
\\midrule
{agg_str}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\end{{document}}
"""
    with open(tbl_tex_path, "w") as f:
        f.write(table_tex_content)
    print(f"Saved: {tbl_tex_path}")

    if has_pdflatex:
        try:
            subprocess.run(
                [pdflatex_bin, "-interaction=nonstopmode", "table_metrics_by_env.tex"],
                cwd=latex_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if os.path.exists(tbl_pdf_path):
                print(f"Compiled: {tbl_pdf_path}")
        except Exception as e:
            print(f"Warning: pdflatex failed for {tbl_tex_path}: {e}")

    # 4. LaTeX Booktabs Executive Summary Table
    summary_tex_path = os.path.join(latex_dir, "table_aggregated_summary.tex")
    summary_pdf_path = os.path.join(latex_dir, "table_aggregated_summary.pdf")

    naive_mse = data_table["mse_samp"]["naive"][-1]
    summary_rows = []
    for v in variants:
        cossim = data_table["rho_samp"][v][-1]
        var = data_table["var_samp"][v][-1]
        mse = data_table["mse_samp"][v][-1]
        mse_ratio = (mse / naive_mse) if naive_mse > 0 else 1.0
        summary_rows.append(
            f"{VAR_LABELS_TEX[v]} & {fmt_cos(cossim)} & {fmt_num(var)} & {fmt_num(mse)} & {mse_ratio:.2f}$\\times$ \\\\"
        )
    summary_rows_code = "\n".join(summary_rows)

    summary_tex_content = f"""\\documentclass[preview,border=6pt]{{standalone}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}

\\begin{{document}}
\\begin{{table}}[htbp]
\\centering
\\caption{{Aggregated Sampled $E(\\lambda)$ Performance Summary across 8 Benchmark Environments ($\\lambda=0.8, \\gamma=0.99$).}}
\\vspace{{4pt}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Formulation}} & \\textbf{{Cosine Sim}} $\\cos(\\bar{{g}}, g^*) \\uparrow$ & \\textbf{{Variance}} $\\mathrm{{Var}}(\\bar{{g}}) \\downarrow$ & \\textbf{{MSE}} $||\\bar{{g}} - g^*||^2 \\downarrow$ & \\textbf{{MSE vs Naive}} \\\\
\\midrule
{summary_rows_code}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\end{{document}}
"""
    with open(summary_tex_path, "w") as f:
        f.write(summary_tex_content)
    print(f"Saved: {summary_tex_path}")

    if has_pdflatex:
        try:
            subprocess.run(
                [pdflatex_bin, "-interaction=nonstopmode", "table_aggregated_summary.tex"],
                cwd=latex_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if os.path.exists(summary_pdf_path):
                print(f"Compiled: {summary_pdf_path}")
        except Exception as e:
            print(f"Warning: pdflatex failed for {summary_tex_path}: {e}")


def export_summary_table(all_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], out_dir: str):
    envs = [e for e in ALL_ENVS if e in all_data]
    if not envs:
        return

    rows = []
    variants = ["naive", "aux", "aux_mc", "aux_full_mc", "fwd"]
    metric_keys = [
        "exact_e_loss",
        "loss_samp_naive", "loss_samp_aux", "loss_samp_aux_mc", "loss_samp_aux_full_mc", "loss_samp_fwd",
        "var_samp_naive", "var_samp_aux", "var_samp_aux_mc", "var_samp_aux_full_mc", "var_samp_fwd",
        "mse_samp_naive", "mse_samp_aux", "mse_samp_aux_mc", "mse_samp_aux_full_mc", "mse_samp_fwd",
        "rho_samp_naive_exact", "rho_samp_aux_exact", "rho_samp_aux_mc_exact", "rho_samp_aux_full_mc_exact", "rho_samp_fwd_exact",
    ]

    for env in envs:
        row = {"Environment": env}
        for k in metric_keys:
            if k in all_data[env]:
                m, _ = all_data[env][k]
                m_arr = np.asarray(m)
                row[f"{k}_mean"] = float(np.nanmean(m_arr))
                row[f"{k}_final"] = float(m_arr[-1]) if len(m_arr) > 0 else np.nan
        # Cosine AUC
        for var in variants:
            k = f"rho_samp_{var}_exact"
            if k in all_data[env]:
                m, _ = all_data[env][k]
                row[f"{k}_auc"] = float(np.nanmean(m))
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "summary_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/sampled_E_lambda_test")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir if args.out_dir is not None else results_dir
    os.makedirs(out_dir, exist_ok=True)

    all_data = {}
    for env in ALL_ENVS:
        m_list = load_env_metrics(results_dir, env)
        if m_list:
            agg = aggregate_seeds(m_list)
            if agg:
                all_data[env] = agg

    if not all_data:
        subdirs = sorted(glob.glob(os.path.join(results_dir, "lambda_*")))
        if subdirs:
            print(f"Found lambda subdirectories: {subdirs}. Processing each...")
            for sdir in subdirs:
                sub_out = os.path.join(out_dir, os.path.basename(sdir))
                os.makedirs(sub_out, exist_ok=True)
                sub_data = {}
                for env in ALL_ENVS:
                    m_list = load_env_metrics(sdir, env)
                    if m_list:
                        agg = aggregate_seeds(m_list)
                        if agg:
                            sub_data[env] = agg
                if sub_data:
                    print(f"--- Processing {os.path.basename(sdir)} with {len(sub_data)} environments ---")
                    # 1. Existing combined bar chart
                    plot_aggregate_bar_chart(sub_data, sub_out)
                    # 2. Individual bar charts (PNG + PDF)
                    plot_individual_bar_charts(sub_data, sub_out)
                    # 3. Time series curves
                    plot_metric_time_series(sub_data, "rho_samp", r"Cosine Similarity to Exact $\nabla E(\lambda)$", r"$\cos(\bar{g}, g^*)$", "gradient_cosine_curves.png", sub_out)
                    plot_metric_time_series(sub_data, "var_samp", r"Gradient Variance $\mathrm{Var}(\bar{g})$", "Variance", "gradient_variance_curves.png", sub_out, log_scale=True)
                    plot_metric_time_series(sub_data, "mse_samp", r"Gradient MSE $||\bar{g} - g^*||^2$", "MSE", "gradient_mse_curves.png", sub_out, log_scale=True)
                    plot_metric_time_series(sub_data, "loss_samp", r"Sampled $E(\lambda)$ Loss", "Loss", "loss_curves.png", sub_out, log_scale=True)
                    # 4. LaTeX plots and booktabs tables
                    export_latex_plots_and_tables(sub_data, sub_out)
                    # 5. CSV summary table
                    export_summary_table(sub_data, sub_out)
            return

        print(f"No metric files found in {results_dir}.")
        return

    print(f"Loaded results for {len(all_data)} environments: {list(all_data.keys())}")
    plot_aggregate_bar_chart(all_data, out_dir)
    plot_individual_bar_charts(all_data, out_dir)
    plot_metric_time_series(all_data, "rho_samp", r"Cosine Similarity to Exact $\nabla E(\lambda)$", r"$\cos(\bar{g}, g^*)$", "gradient_cosine_curves.png", out_dir)
    plot_metric_time_series(all_data, "var_samp", r"Gradient Variance $\mathrm{Var}(\bar{g})$", "Variance", "gradient_variance_curves.png", out_dir, log_scale=True)
    plot_metric_time_series(all_data, "mse_samp", r"Gradient MSE $||\bar{g} - g^*||^2$", "MSE", "gradient_mse_curves.png", out_dir, log_scale=True)
    plot_metric_time_series(all_data, "loss_samp", r"Sampled $E(\lambda)$ Loss", "Loss", "loss_curves.png", out_dir, log_scale=True)
    export_latex_plots_and_tables(all_data, out_dir)
    export_summary_table(all_data, out_dir)


if __name__ == "__main__":
    main()

