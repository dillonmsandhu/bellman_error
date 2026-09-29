# EightRooms Benchmark: [Continuous vs Discrete] x [Sparse vs Dense]

This suite evaluates and compares **Ground Truth PPO** and **Exact E(λ) PPO** across $\lambda \in \{0.0, 0.8, 0.95, 0.99, 1.0\}$ on the full 2x2 factorial grid of EightRooms:

1. **Discrete Sparse**: `EightRooms-misc` (classic discrete 25x13 gridworld, sparse +1.0 goal reward)
2. **Continuous Sparse**: `continuous-eightrooms` (continuous control with $[-1, 1]^2$ actions, identical 8-rooms geometry, sparse goal reward)
3. **Discrete Dense**: `EightRooms-Dense` (discrete 8 rooms with Potential-Based Reward Shaping $\Phi(s) = -0.03125 \cdot \text{GeodesicDist}(s, \text{goal})$)
4. **Continuous Dense**: `continuous-eightrooms-dense` (continuous control with bilinear interpolation of the exact geodesic distance potential field)

---

## Experiment Structure

- **Sub-Jobs**: Exactly 1 sub-job per environment (`#SBATCH --array=0-3`).
  - Task `0`: `EightRooms-misc` (discrete sparse)
  - Task `1`: `continuous-eightrooms` (continuous sparse)
  - Task `2`: `EightRooms-Dense` (discrete dense)
  - Task `3`: `continuous-eightrooms-dense` (continuous dense)
- **Exact Settings Enforced**:
  - `NUM_TIMESTEPS=1000` (or `TOTAL_TIMESTEPS=1000`)
  - `NUM_EPOCHS=1`
  - `NUM_MINIBATCHES=1`
  - `N_SEEDS=8`
- **Algorithms / Runs per Sub-Job** (6 total):
  1. `Ground Truth PPO` (exact value oracle baseline)
  2. `Exact E(λ=0.0) PPO`
  3. `Exact E(λ=0.8) PPO`
  4. `Exact E(λ=0.95) PPO`
  5. `Exact E(λ=0.99) PPO`
  6. `Exact E(λ=1.0) PPO`

---

## How to Run

### 1. Submit Full SLURM Array Job (Recommended)

Submit the self-contained array job directly:
```bash
sbatch scripts/eightrooms_lambda_sweep/run_slurm_eightrooms_array.sh
```

Or launch with the dispatcher (schedules a chained plotting job):
```bash
./scripts/eightrooms_lambda_sweep/submit_eightrooms_array.sh
```

Preview commands with dry run:
```bash
./scripts/eightrooms_lambda_sweep/submit_eightrooms_array.sh --dry-run
```

### 2. Run Individual Sub-Jobs Locally

```bash
# Task 0: Discrete Sparse
./scripts/eightrooms_lambda_sweep/run_slurm_eightrooms_array.sh 0

# Task 1: Continuous Sparse
./scripts/eightrooms_lambda_sweep/run_slurm_eightrooms_array.sh 1

# Task 2: Discrete Dense
./scripts/eightrooms_lambda_sweep/run_slurm_eightrooms_array.sh 2

# Task 3: Continuous Dense
./scripts/eightrooms_lambda_sweep/run_slurm_eightrooms_array.sh 3
```

### 3. Generate Learning Curves Plot

```bash
python scripts/eightrooms_lambda_sweep/plot_eightrooms_learning_curves.py --sweep-id <SWEEP_ID>
```
Default layout is 2x2. To output a 1x4 horizontal strip instead:
```bash
python scripts/eightrooms_lambda_sweep/plot_eightrooms_learning_curves.py --sweep-id <SWEEP_ID> --layout 1x4
```

---

## Output Artifacts

Outputs are saved under `results/eightrooms_sweep/<SWEEP_ID>/plots/`:
- `learning_curves_V_start.pdf`: Vector PDF showing the 4-panel comparison with SEM error bands across 8 seeds.
- `summary_V_start.csv`: Summary table with Final Window Mean, SEM, and AUC for all 6 conditions.
