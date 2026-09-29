# FourRooms & EightRooms Benchmark: Ground Truth vs. Exact E(λ) PPO

This suite evaluates and compares **Ground Truth PPO** and **Exact E(λ) PPO** across $\lambda \in \{0.0, 0.8, 0.95, 1.0\}$ on four benchmark environments:

1. **Regular 4 Rooms** (`FourRooms-misc`): Classic discrete gridworld (13x13), sparse reward.
2. **Continuous 4 Rooms** (`continuous-fourrooms`): Continuous control ($[-1, 1]^2$ actions) with identical 4-rooms geometry and sparse goal reward (+1.0 at goal, 0 elsewhere).
3. **Continuous Dense 4 Rooms** (`continuous-fourrooms-dense`): Continuous control with Potential-Based Reward Shaping (PBRS) using continuous bilinear interpolation of the shortest-path geodesic distance grid $\Phi(s) = -0.03125 \cdot \text{GeodesicDist}(s, \text{goal})$.
4. **Discrete 8 Rooms Dense** (`EightRooms-Dense`): Classic discrete 8-rooms gridworld (25x13) with the exact same dense potential shaping $\Phi(s) = -0.03125 \cdot \text{GeodesicDist}(s, \text{goal})$.

---

## Experiment Structure

- **Sub-Jobs**: Exactly 1 sub-job per environment (`#SBATCH --array=0-3`).
  - Task `0`: `FourRooms-misc`
  - Task `1`: `continuous-fourrooms`
  - Task `2`: `continuous-fourrooms-dense`
  - Task `3`: `EightRooms-Dense`
- **Exact Settings Enforced**:
  - `NUM_TIMESTEPS=1000` (or `TOTAL_TIMESTEPS=1000`)
  - `NUM_EPOCHS=1`
  - `NUM_MINIBATCHES=1`
  - `N_SEEDS=8`
- **Runs per Sub-Job**: 5 distinct runs (not hyperparameter tuning):
  1. `Ground Truth PPO`
  2. `Exact E(λ=0.0) PPO`
  3. `Exact E(λ=0.8) PPO`
  4. `Exact E(λ=0.95) PPO`
  5. `Exact E(λ=1.0) PPO`

---

## How to Run

### 1. Submit Full SLURM Array Job (Recommended)

To launch all 4 environments across 8 seeds and exact settings:
```bash
./scripts/fourrooms_lambda_sweep/submit_fourrooms_array.sh
```

Preview commands without submitting:
```bash
./scripts/fourrooms_lambda_sweep/submit_fourrooms_array.sh --dry-run
```

Or submit directly via `sbatch`:
```bash
sbatch scripts/fourrooms_lambda_sweep/run_slurm_fourrooms_array.sh
```

### 2. Run Individual Sub-Jobs Locally

```bash
# Task 0: Regular 4 rooms
./scripts/fourrooms_lambda_sweep/run_slurm_fourrooms_array.sh 0

# Task 1: Continuous 4 rooms
./scripts/fourrooms_lambda_sweep/run_slurm_fourrooms_array.sh 1

# Task 2: Continuous dense 4 rooms
./scripts/fourrooms_lambda_sweep/run_slurm_fourrooms_array.sh 2

# Task 3: Discrete 8 rooms dense
./scripts/fourrooms_lambda_sweep/run_slurm_fourrooms_array.sh 3
```

### 3. Generate Learning Curves Plot

```bash
python scripts/fourrooms_lambda_sweep/plot_fourrooms_learning_curves.py --sweep-id <SWEEP_ID>
```

---

## Output Artifacts

Outputs are saved under `results/fourrooms_sweep/<SWEEP_ID>/plots/`:
- `learning_curves_V_start.pdf`: Vector PDF showing the 4-panel comparison with SEM error bands across the 8 seeds.
- `summary_V_start.csv`: Summary table with Final Window Mean, SEM, and AUC for all conditions.
