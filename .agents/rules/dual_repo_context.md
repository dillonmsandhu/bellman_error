---
trigger: always_on
---

# Dual-Repository Project Architecture: bellman_error & E_min

## Overview & Strategic Roles
This research project spans two complementary repositories operating on the same underlying Reinforcement Learning research:

### 1. `bellman_error` (Research & Development Sandbox)
- **Path**: `/Users/dillonsandhu/Documents/Research/bellman_error`
- **Role**: Core R&D repository. Focused on algorithmic formulation, mathematical correctness, debugging, and deep diagnostics.
- **Environment Scope**: Tabular environments, exact transition matrices, analytical Bellman error metrics, and fast CPU iterations.
- **Key Breakthrough / Algorithmic Asset**: Contains the strong, validated implementation of the **E-minimization algorithm with auxiliary value network (`v_aux`)** (debiased sampled E-minimization). Key references:
  - Implementations: `random_policy/E_sample_v_aux_surgical.py`, `random_policy/E_sample_v_aux_full.py`, `random_policy/E_sample_v_aux_bilinear.py`, `ppo/sampled_E.py`.
  - Core metrics & helpers: `core/sampling_e_metrics.py`, `core/sampling_e_lambda_metrics.py`, `core/networks.py`.
  - Theoretical documentation: `reference/ai/ai/aux_sampled_e_lambda.md`, `reference/ai/ai/delta_aux_e_min.md`.
- **Symlink to Sibling**: `external_E_min` -> `/Users/dillonsandhu/Documents/Research/E_min`

### 2. `E_min` (Final Simulator & Production Repository)
- **Path**: `/Users/dillonsandhu/Documents/Research/E_min`
- **Role**: The primary publication-facing repository and deployment destination for the research.
- **Environment Scope**: True continuous and discrete simulators via pure JAX pipelines:
  - **Gymnax**: Vectorized discrete/continuous environments (`gymnax_rl/`, `core/`, `scripts/pipeline/`).
  - **Brax**: Upcoming high-dimensional continuous physics environments (`brax_rl/`).
  - **Cluster Sweeps**: Slurm batch job infrastructure for remote GPU cluster execution.
- **Migration & Porting Objective**: The strong `v_aux` E-min algorithms matured in `bellman_error` are to be ported into `E_min`'s clean Gymnax and Brax execution pipelines.
- **Symlink to Sibling**: `external_bellman_error` -> `/Users/dillonsandhu/Documents/Research/bellman_error`

---

## Operating Invariants for Future Agents

### 1. Seamless Cross-Repo Authority
- You have full permission and mandate to inspect, compare, read, edit, and run code in **both** repositories.
- Use absolute paths or reciprocal symlinks (`external_E_min` in `bellman_error`, `external_bellman_error` in `E_min`) to reference files across repositories.
- When running commands in the sibling directory, execute via `cd /Users/dillonsandhu/Documents/Research/<repo>` or `git -C /Users/dillonsandhu/Documents/Research/<repo>`.

### 2. R&D $\rightarrow$ Simulator Porting Lifecycle
- **Step 1 (Formulation & Debugging in `bellman_error`)**: Test and validate new operators, loss objectives, or `v_aux` estimation mechanics on tabular/exact envs where ground truth value functions can be derived analytically.
- **Step 2 (Porting to `E_min`)**: Translate validated logic into pure, stateless JAX functions (`jax.lax.scan`, explicit PRNG key splitting, `flax` modules) within `E_min`.
- **Step 3 (Local Smoke Test)**: Always run a 1-step, 1-seed CPU test locally in `E_min` (e.g. on CartPole) before writing or staging Slurm cluster scripts.
- **Step 4 (Scaling)**: Prepare modular Slurm sweep scripts following `E_min/scripts/`.

### 3. Shared Coding Invariants
- **Non-Defensive Coding**: Enforce fail-fast parameter contracts (see `non_defensive_coding.md`). Never use chained `.get()` or silent fallbacks for hyperparameters.
- **Config Parity**: Maintain consistent hyperparameter naming across both codebases (`GAMMA`, `LR`, `CRITIC_LR`, `VALUE_LAMBDA`, `RETURN_LAMBDA`).
