import os
import json
import cloudpickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from core.utils import load_run_data, load_run_data_from_path

def get_latest_run_path(base_tuning_dir):
    """Finds the latest timestamped directory and environment name under a tuning or results directory."""
    if not os.path.exists(base_tuning_dir):
        raise FileNotFoundError(f"Directory not found: {base_tuning_dir}")
    timestamps = sorted([d for d in os.listdir(base_tuning_dir) if os.path.isdir(os.path.join(base_tuning_dir, d))])
    if not timestamps:
        raise FileNotFoundError(f"No run timestamps found in {base_tuning_dir}")
    latest_ts = timestamps[-1]
    ts_path = os.path.join(base_tuning_dir, latest_ts)
    envs = [e for e in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, e))]
    if not envs:
        raise FileNotFoundError(f"No environment folders found in {ts_path}")
    env_name = envs[0]
    return latest_ts, env_name, os.path.join(base_tuning_dir, latest_ts, env_name)

def load_sweep_run(tuning_base_dir):
    """Loads config, metrics, and tuning summary for a sweep/tuning run."""
    ts, env_name, run_dir = get_latest_run_path(tuning_base_dir)
    config, metrics = load_run_data(ts, env_name, results_base_path=tuning_base_dir)
    
    summary_path = os.path.join(run_dir, "tuning_summary.csv")
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    
    print(f"Loaded sweep run from {run_dir}")
    return config, metrics, summary_df

def load_sweep_data_from_path(run_dir):
    """Loads config, metrics, and tuning summary for a sweep/tuning run."""
    config, metrics = load_run_data_from_path(run_dir)
    
    summary_path = os.path.join(run_dir, "tuning_summary.csv")
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    
    print(f"Loaded sweep run from {run_dir}")
    return config, metrics, summary_df

def plot_mean_std_curve(data_tensor, steps_per_pi=1, metric_name="Metric", ylabel="Value", log_scale=True, label_prefix="Run", tuning_summary=None):
    """
    Plots a single run's mean trajectory with a standard deviation band (± 1 std) across seeds.
    """
    plot_multi_mean_std_curves(
        {label_prefix: data_tensor},
        steps_per_pi=steps_per_pi,
        metric_name=metric_name,
        ylabel=ylabel,
        log_scale=log_scale,
        tuning_summaries={label_prefix: tuning_summary} if tuning_summary is not None else None
    )

def plot_multi_mean_std_curves(runs_dict, steps_per_pi=1, metric_name="Metric", ylabel="Value", log_scale=True, tuning_summaries=None):
    """
    Plots multiple runs' mean trajectories with standard deviation bands (± 1 std) on the same plot for comparison.
    
    Args:
        runs_dict: Dictionary mapping run labels (str) to data tensors (array-like of shape 
                   (n_seeds, time_steps) or (n_combos, n_seeds, time_steps)).
        steps_per_pi: Number of environment steps between data points.
        metric_name: Title/label for the metric.
        ylabel: Y-axis label.
        log_scale: Whether to use log scale on y-axis.
        tuning_summaries: Optional dict mapping run labels to tuning summary DataFrames.
    """
    plt.figure(figsize=(10, 6))

    for label, data_tensor in runs_dict.items():
        arr = np.array(data_tensor)
        if arr.ndim == 3:
            if tuning_summaries is not None and label in tuning_summaries and tuning_summaries[label] is not None:
                summary_df = tuning_summaries[label]
                best_idx = int(summary_df.iloc[0]['config_idx'])
                lr = summary_df.loc[summary_df['config_idx'] == best_idx]['LR'].item()
                print(f"[{label}] Using best combo index {best_idx} from tuning summary.")
            else:
                print(f"label {label} not in tuning summaries")
                # If multiple combos and no summary provided, take the best combo (lowest final value across seeds)
                final_means = arr[:, :, -1].mean(axis=1)
                best_idx = int(np.argmin(final_means))
                print(f"[{label}] Using best combo index {best_idx} from 3D tensor.")
            arr = arr[best_idx]
        
        if arr.ndim != 2:
            raise ValueError(f"[{label}] Expected 2D array (n_seeds, time_steps), got shape {arr.shape}")

        n_seeds, time_steps = arr.shape
        print('number of seeds:', n_seeds)
        x = [i * steps_per_pi for i in range(time_steps)]
        
        mean_curve = arr.mean(axis=0)
        std_curve = arr.std(axis=0)
        # Median with 16th and 84th percentiles (equivalent to ±1 std for normal data)
        # median_curve = np.median(arr, axis=0)
        # lower_bound = np.percentile(arr, 16, axis=0)
        # upper_bound = np.percentile(arr, 84, axis=0)

        # line, = plt.plot(x, median_curve, label=f"{label} (Median)", linewidth=2)
        # plt.fill_between(x, lower_bound, upper_bound, color=line.get_color(), alpha=0.2)

        # line, = plt.plot(x, mean_curve, label=f"{label} (Mean)", linewidth=2)
        # color = line.get_color()
        # plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2, label=f"{label} (±1 std)")

        # Geometric mean and multiplicative standard deviation
        log_arr = np.log(arr)
        log_mean = np.mean(log_arr, axis=0)
        log_std = np.std(log_arr, axis=0)

        geom_mean = np.exp(log_mean)
        lower_bound = np.exp(log_mean - log_std)
        upper_bound = np.exp(log_mean + log_std)
        try:
            line, = plt.plot(x, geom_mean, label=f"{label} (LR = {lr})", linewidth=2)
        except:
            line, = plt.plot(x, geom_mean, label=f"{label}", linewidth=2)
        plt.fill_between(x, lower_bound, upper_bound, color=line.get_color(), alpha=0.2)

    if log_scale:
        plt.yscale('log')
    plt.xlabel("Gradient Update Steps")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} Comparison over Training Course")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_generalization_heatmaps(metrics, env_name="Environment", show_start=True):
    """
    Plots heatmaps for NTK, Centered NTK, Jacobian singular vectors, and Feature singular vectors
    to help analyze and compare generalization.
    """
    # 1. NTK Matrix Heatmap (using final state eNTK if available)
    if "eNTK" in metrics:
        eNTK_data = np.array(metrics["eNTK"])
        # If batched over combo/seed, extract final
        while eNTK_data.ndim > 2:
            eNTK_data = eNTK_data[0]
        
        plt.figure(figsize=(8, 6))
        plt.matshow(eNTK_data, cmap="viridis")
        plt.colorbar(label="NTK Value")
        plt.title(f"{env_name} - Empirical NTK Matrix (eNTK)")
        plt.xlabel("State Index")
        plt.ylabel("State Index")
        plt.tight_layout()
        plt.show()

    # 2. Centered NTK Matrix Heatmap (Gradient Covariance Matrix)
    if "gradient_covariance_matrix" in metrics:
        cov_data = np.array(metrics["gradient_covariance_matrix"])
        # If batched over combo/seed, extract final
        while cov_data.ndim > 2:
            cov_data = cov_data[0]
        
        plt.figure(figsize=(8, 6))
        plt.matshow(cov_data, cmap="viridis")
        plt.colorbar(label="Covariance Value")
        plt.title(f"{env_name} - Centered NTK (Gradient Covariance Matrix)")
        plt.xlabel("State Index")
        plt.ylabel("State Index")
        plt.tight_layout()
        plt.show()

    # 3. Jacobian Top Singular Vectors
    if "Jacobian_top_singular_vectors" in metrics:
        j_svs = np.array(metrics["Jacobian_top_singular_vectors"])
        # If batched over combo/seed, extract final
        while j_svs.ndim > 4:
            j_svs = j_svs[0]
        # typically shape (time_steps, 5, H, W) or (time_steps, n_components, H, W)
        # we focus on the end of training: j_svs[-1]
        stack = j_svs[-1] if j_svs.ndim == 4 else j_svs
        n_components = min(5, len(stack))
        
        fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
        fig.suptitle(f"{env_name} - Top {n_components} Jacobian Singular Vectors", fontsize=16)
        
        if n_components == 1:
            axes = [axes]
            
        for i in range(n_components):
            grid = stack[i]
            max_abs = np.max(np.abs(grid))
            if max_abs == 0:
                max_abs = 1.0
            
            ax = axes[i]
            im = ax.matshow(grid, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
            ax.set_title(f"Component {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    # 4. Feature Top Singular Vectors
    if "feature_top_singular_vectors" in metrics:
        f_svs = np.array(metrics["feature_top_singular_vectors"])
        # If batched over combo/seed, extract final
        while f_svs.ndim > 4:
            f_svs = f_svs[0]
        stack = f_svs[-1] if f_svs.ndim == 4 else f_svs
        n_components = min(5, len(stack))
        
        fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
        fig.suptitle(f"{env_name} - Top {n_components} Feature Singular Vectors", fontsize=16)
        
        if n_components == 1:
            axes = [axes]
            
        for i in range(n_components):
            grid = stack[i]
            max_abs = np.max(np.abs(grid))
            if max_abs == 0:
                max_abs = 1.0
            
            ax = axes[i]
            im = ax.matshow(grid, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
            ax.set_title(f"Component {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

# Typo-tolerant alias as explicitly requested
plot_geneneralizatio_heatmaps = plot_generalization_heatmaps

def plot_multi_spectra(metrics_dict, env_name="Environment", show_start=True):
    """
    Plots spectra for Jacobian and Features comparing multiple runs (e.g. TD vs MC).
    metrics_dict: Dictionary mapping run labels (str) to metrics dictionaries.
    """
    colors = plt.cm.tab10.colors
    
    # Jacobian
    plt.figure(figsize=(8, 5))
    has_jacobian = False
    for i, (label, metrics) in enumerate(metrics_dict.items()):
        if "jacobian_singular_values" in metrics:
            has_jacobian = True
            color = colors[i % len(colors)]
            j_sv = np.array(metrics["jacobian_singular_values"])
            while j_sv.ndim > 2:
                j_sv = j_sv[0]
            start_sv = j_sv[0] if j_sv.ndim == 2 else j_sv
            end_sv = j_sv[-1] if j_sv.ndim == 2 else j_sv
            
            if show_start and j_sv.ndim == 2:
                # lighter version of the color for start
                light_color = tuple(min(1.0, c + 0.5 * (1.0 - c)) for c in color[:3])
                plt.plot(np.arange(1, len(start_sv)+1), start_sv, linestyle='--', color=light_color, alpha=0.7, label=f'{label} Start')
            plt.plot(np.arange(1, len(end_sv)+1), end_sv, color=color, label=f'{label} End')
            
    if has_jacobian:
        plt.yscale('log')
        plt.xlabel("Singular Value Index")
        plt.ylabel("Singular Value Magnitude (Log Scale)")
        plt.title(f"{env_name} - Jacobian Singular Value Spectrum Comparison")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    else:
        plt.close()

    # Features
    plt.figure(figsize=(8, 5))
    has_features = False
    for i, (label, metrics) in enumerate(metrics_dict.items()):
        if "feature_singular_values" in metrics:
            has_features = True
            color = colors[i % len(colors)]
            f_sv = np.array(metrics["feature_singular_values"])
            while f_sv.ndim > 2:
                f_sv = f_sv[0]
            start_sv = f_sv[0] if f_sv.ndim == 2 else f_sv
            end_sv = f_sv[-1] if f_sv.ndim == 2 else f_sv
            
            if show_start and f_sv.ndim == 2:
                light_color = tuple(min(1.0, c + 0.5 * (1.0 - c)) for c in color[:3])
                plt.plot(np.arange(1, len(start_sv)+1), start_sv, linestyle='--', color=light_color, alpha=0.7, label=f'{label} Start')
            plt.plot(np.arange(1, len(end_sv)+1), end_sv, color=color, label=f'{label} End')

    if has_features:
        plt.yscale('log')
        plt.xlabel("Singular Value Index")
        plt.ylabel("Singular Value Magnitude (Log Scale)")
        plt.title(f"{env_name} - Feature Singular Value Spectrum Comparison")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    else:
        plt.close()
