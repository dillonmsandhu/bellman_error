---
name: slurm-sweeps-and-runs
description: Create, submit, and manage SLURM batch arrays, hyperparameter sweeps, and large-scale training runs on the cluster. Enforces cluster partition rules (compsci-gpu, gpu:a5000:1), self-contained script architecture, SLURM_SUBMIT_DIR working directory handling, multi-environment arrays, chained post-processing dependencies, and python interpreter resolution.
---

# SLURM Sweeps and Large-Scale Cluster Runs

This skill guides authoring, configuring, and launching SLURM job arrays, parameter sweeps, and large training experiments on the Duke CompSci GPU cluster.

---

## 1. Cluster Hardware & SLURM Directives

Always configure the batch headers to target the designated cluster hardware:

```bash
#!/bin/bash
#SBATCH --job-name=<JOB_NAME>
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-<N-1>
```

### Resource Guidelines:
- **Partition**: `compsci-gpu` for GPU-accelerated jobs; `compsci` for CPU-only post-processing/plotting jobs.
- **GPU GRES**: `gpu:a5000:1` (NVIDIA RTX A5000 24GB).
- **CPUs & Memory**: 4 CPUs and 32GB RAM per task is the standard sweet spot.
- **Time Limits**: `04:00:00` for standard arrays; `12:00:00` for extensive parameter grids.

---

## 2. Critical SLURM Gotchas

### Gotcha A: `SLURM_SUBMIT_DIR` vs `${BASH_SOURCE[0]}`
When SLURM executes a script submitted with `sbatch`, it copies the script to a daemon spool directory (e.g. `/var/lib/slurm/slurmd/jobXXXX/slurm_script`).
* **Never** rely solely on `$(dirname "${BASH_SOURCE[0]}")` to find the repo root inside a SLURM job.
* **Always** use `$SLURM_SUBMIT_DIR` (the directory from which `sbatch` was called):

```bash
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Gotcha B: Prefer Self-Contained Scripts Over Multi-File Workers
Avoid having `run_slurm.sh` call a secondary `worker.sh` via relative path.
* Putting the worker logic directly inside `run_slurm_array.sh` eliminates relative path errors across cluster nodes.
* It allows running a single sub-job locally for debugging:
  `./scripts/my_sweep/run_slurm_array.sh 0`

### Gotcha C: Python Environment Resolution
Use the standard fallback chain to support execution on both cluster nodes and local development machines:

```bash
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi
```

---

## 3. Self-Contained Array Template

```bash
#!/bin/bash
#SBATCH --job-name=my_array_job
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3

set -e
mkdir -p slurm

# Working directory resolution
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Python resolution
if [ -f "/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python" ]; then
    PYTHON="/home/users/ds541/.pyenv/versions/3.10.15/envs/gymnax/bin/python"
elif [ -f "/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python" ]; then
    PYTHON="/Users/dillonsandhu/.pyenv/versions/gymnax/bin/python"
else
    PYTHON="python"
fi

ENVS=(
    "FourRooms-misc"
    "continuous-fourrooms"
    "continuous-fourrooms-dense"
    "EightRooms-Dense"
)

# Allow manual index via $1 or SLURM_ARRAY_TASK_ID
TARGET_ARG=${1:-$SLURM_ARRAY_TASK_ID}
if [ -z "$TARGET_ARG" ]; then
    echo "No task index provided; running all tasks sequentially..."
    for task_idx in $(seq 0 $((${#ENVS[@]} - 1))); do
        "$0" "$task_idx"
    done
    exit 0
fi

if [[ "$TARGET_ARG" =~ ^[0-9]+$ ]]; then
    ENV_NAME="${ENVS[$TARGET_ARG]}"
else
    ENV_NAME="$TARGET_ARG"
fi

# Shared Sweep ID
if [ -z "$SWEEP_ID" ]; then
    if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
        SWEEP_ID="sweep_${SLURM_ARRAY_JOB_ID}"
    else
        SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
    fi
fi

# Run workload for this specific environment...
```

---

## 4. Submission Dispatcher & Chained Post-Processing

To automatically generate plots/diagnostics as soon as all array tasks complete, provide a dispatcher script `submit_*.sh` that chains a dependent job via `--dependency=afterany:<ARRAY_JOB_ID>`:

```bash
# 1. Submit Array Job
SBATCH_ARRAY_CMD="sbatch --parsable \
    --export=ALL,SWEEP_ID=$SWEEP_ID,SLURM_SUBMIT_DIR=$REPO_ROOT \
    \"$ARRAY_SCRIPT\""
ARRAY_JOB_ID=$(eval "$SBATCH_ARRAY_CMD")

# 2. Submit Chained Post-Processing (Runs after array finishes)
sbatch \
    --dependency=afterany:$ARRAY_JOB_ID \
    --job-name="plot_${SWEEP_ID}" \
    --output="slurm/%j_plot_${SWEEP_ID}.out" \
    --time=00:30:00 \
    --partition=compsci \
    --wrap="$PYTHON '$PLOT_SCRIPT' --sweep-id '$SWEEP_ID'"
```

---

## 5. Exact vs. Sampled Hyperparameter Standards

| Setting | Exact PPO / Exact Evaluation | Sampled PPO / Rollouts |
|---|---|---|
| `TOTAL_TIMESTEPS` / `NUM_TIMESTEPS` | `1000` (iterations over full $P$) | `1000000` (environment steps) |
| `NUM_EPOCHS` | `1` | `4` |
| `NUM_MINIBATCHES` | `1` | `4` |
| `MINIBATCH_SIZE` | `1` (or full batch) | `1024` |
| `NUM_ENVS` x `NUM_STEPS` | `1` x `1` | `128` x `128` |
| `N_SEEDS` | `8` | `5` - `8` |

---

## 6. Plotting & Publication Guidelines

- **Vector PDF Only**: Export strictly to `.pdf` with `pdf.fonttype = 42` and `ps.fonttype = 42` (TrueType embedding). Avoid raster PNG unless specifically requested.
- **Error Bands**: Display Mean $\pm$ SEM across seeds (`sem = std / np.sqrt(n_seeds)`).
- **Summary Metrics**: Always export an accompanying `summary_<metric>.csv` containing the final window mean, SEM, and Area Under the Curve (AUC) for each condition.
