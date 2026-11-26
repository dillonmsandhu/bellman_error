# ICML 2027 Hyperparameter Tuning & Evaluation Protocol

This document defines the fair, literature-standard **two-stage tuning and held-out evaluation pipeline** for comparing value-learning critics across our three Deep RL benchmark suites: **Brax**, **MJX Playground Locomotion**, and **Octax**.

---

## 0. Mandatory Pre-Sweep Parity Checklist (Fix Before Any Final Runs)

Before running or reporting final benchmark numbers, verify **strict apples-to-apples parity** across all critic methods:

1. **Loss Scaling (`VF_COEF`) Parity:** Ensure $\mathrm{TD}(\lambda)$, $E(\lambda)$, and TDRC use the **exact same `VF_COEF`** (e.g., `0.5` or `1.0` for all methods). Even under Adam, a `0.5×` difference changes when `MAX_GRAD_NORM` clipping triggers—and if actor and critic share a single global norm clip, it directly alters the effective actor-to-critic step size ratio.
2. **Value Function Clipping (`VF_CLIP = False`):** Disable PPO's heuristic `VF_CLIP` across **all** critics (regular fitted $\mathrm{TD}(\lambda)$ vs. $E(\lambda)$). Enabling it for TD alone breaks parity; enabling it for $E$ distorts the analytical $E$ gradient. State this explicitly in the paper.
3. **Separate Optimizers & Clipping:** Use separate optimizers and `clip_by_global_norm` for **actor**, **main critic ($v$)**, and **auxiliary critic ($v_{\text{aux}}$ / `aux_loss`)**. Ensuring `aux_loss` has its own dedicated optimizer (verified design choice) guarantees that auxiliary MC gradient norms and clipping never distort or throttle the main critic $v$'s updates.
4. **Identical Normalization & `GAMMA`:** Same wrappers (`NORMALIZE_ENV`, `NORMALIZE_REW`) and exact same `GAMMA` per suite across all methods.
5. **Equal Search Budget:** Every tuned algorithm sweeps the **exact same grid** `(CRITIC_LR, CRITIC_EPOCHS, MAX_GRAD_NORM[, λ])`. Auxiliary-specific knobs (`AUX_VF_COEF = 1.0`, `λ_aux = 1.0`, TDRC `β = 1.0`) are **fixed globally across all suites** so auxiliary-head methods never receive a larger search budget than $\mathrm{TD}(\lambda)$.

---

## 1. Benchmark Suite Lineup (Non-Overlapping)

Drop Gymnax from the main paper (6/11 tasks saturate at 100%, and the exact tabular suite already covers FourRooms/EightRooms/MountainCar/SpaceInvaders). Keep three modern, non-overlapping Deep RL suites:

| Suite | Scope | Tuning Envs | Held-Out Eval Envs |
| :--- | :--- | :--- | :--- |
| **Brax** ($\gamma=0.99$) | 9–10 classic continuous control tasks | **Full suite** (all 9–10 envs) | Full suite (disjoint seeds) |
| **MJX Playground Locomotion** ($\gamma=0.97$) | 5 high-DoF robotics locomotion tasks (`BerkeleyHumanoid`, `G1`, `Go1Getup`, `Go1Handstand`, `Go1JoystickFlatTerrain`) — zero overlap with Brax | **Full suite** (all 5 envs) | Full suite (disjoint seeds) |
| **Octax** | Visual/discrete CNN suite (+ Standard CNN vs. Pooled CNN aliasing test) | **4–5 representative envs** (chosen up front across difficulty/dynamics) | **Full Octax suite** (including held-out envs + disjoint seeds) |

---

## 2. Algorithms: Separately Tuned vs. Inherited Ablations

### Separately Tuned Algorithms (Stage 1)
1. **`online_td0`**: Semi-gradient Online $\mathrm{TD}(0)$.
2. **`e0`**: Sampled $E(0)$ with surgical $v_{\text{aux}}$.
3. **`td_lambda`**: Fitted $\mathrm{TD}(\lambda)$ (standard PPO critic, `VF_CLIP = False`).
4. **`e_lambda`**: Sampled $E(\lambda)$ with surgical $v_{\text{aux}}$.
5. **`tdrc`** *(comparable auxiliary-head baseline)*: Gradient TD with regularized $\delta_{\text{aux}}$ head (same grid as `e0` / `e_lambda`).

### Inherited Ablations (No Separate Tuning)
- **`bsmc`**, **`rot_gated_e`**, **`e_aux`**, **`forward_only`**, **`naive`**: Do **not** run separate tuning sweeps; inherit the written JSON config of the closest tuned base algorithm (`e0.json` or `e_lambda.json`) directly in Stage 2. This is both compute-efficient and conservative (zero extra tuning advantage for ablations).

---

## 3. Two-Stage Automated Pipeline Architecture

### Stage 1: Per-`(Suite, Algorithm)` Tuning & Config Export
- **Execution:** A single Slurm sweep per `(suite, algo)` sweeps the grid on **tuning seeds** (e.g., seeds `0, 1, 2`) across the suite's tuning environments.
- **Independent Normalization Scale:** Normalize each environment's AUC return against fixed per-environment reference bounds `(R_random[e], R_ref[e])` stored in `configs/env_scales/{suite}.json`. This allows each algorithm to be tuned completely independently without waiting on `TD(λ)` and prevents scores from shifting when adding baselines.
- **Chained Auto-Selection (`--dependency=afterok`):** Automatically computes suite-wide IQM of normalized AUC and writes explicit JSON configs (with provenance metadata: `suite`, `algo`, `tuning_envs`, `tuning_seeds`, `iqm_score`, `grid`):
  - **Global Suite Winner:** `configs/tuned/{suite}/{algo}.json` (`argmax` over the full grid including $\lambda$).
  - **Per-$\lambda$ Suite Winners** (for `td_lambda` and `e_lambda`, zero extra GPU runs): `configs/tuned/{suite}/{algo}_lam{lam}.json` for each $\lambda \in \{0.0, 0.4, 0.8, 0.95\}$, independently selecting the best `(CRITIC_LR, CRITIC_EPOCHS, MAX_GRAD_NORM)` for each $\lambda$.

### Stage 2: Held-Out Comparative Evaluation & Publication Plots
- **Execution:** A comparative Slurm sweep per `suite` loads the cached JSON configs from `configs/tuned/{suite}/` (failing fast if any required config is missing) and runs **disjoint held-out seeds** (e.g., seeds `100..109` for Brax/Octax; `100..104` for MJX) across the **full suite**.
- **Two Evaluation Modes (both run from cached JSONs):**
  1. **Main Championship:** Global winners (`online_td0.json`, `e0.json`, `td_lambda.json`, `e_lambda.json`, `tdrc.json` + inherited ablations).
  2. **Per-$\lambda$ Robustness Curve:** `td_lambda_lam{0.4,0.8,0.95}.json` vs. `e_lambda_lam{0.4,0.8,0.95}.json` (plus `online_td0` / `e0` at $\lambda = 0$).
- **Chained Post-Processing:** Automatically generates all publication artifacts:
  - Championship summary tables (IQM, Mean Norm, Median, 1% W/T/L on AUC and Final Return).
  - `rliable`-style performance profiles.
  - Per-$\lambda$ IQM vs. $\lambda$ curves paired with online diagnostics (`r_rot,c`, `ρ_c`, `VR_ret`, `1 - R²`).
  - *(Appendix)* Per-environment tuned upper-bound table extracted directly from Stage 1 sweeps.

---

## 4. Handling Actor Hyperparameters & `GAE_LAMBDA`

- **Main Suite Benchmarks:** Use the standard PPO `GAE_LAMBDA` for each suite (document explicitly). Keep `CLIP_EPS`, `ENT_COEF`, `ACTOR_EPOCHS`, `NUM_ENVS`, `NUM_STEPS`, and network architecture fixed to suite defaults.
- **Actor LR (`ACTOR_LR`):** On fast suites (Brax, Octax), optionally include a small symmetric `ACTOR_LR` axis (e.g., `{1e-4, 3e-4}`) in Stage 1 for all algorithms; on 50M–200M-step MJX, fix `ACTOR_LR = 3e-4` (MuJoCo Playground default).
- **Dedicated Critic-Reliance Ablation (1 Suite, e.g., Brax):** Sweep actor `GAE_LAMBDA ∈ {0.0, 0.4, 0.8, 0.95}` using the Stage 1 winning critic configs to cleanly show how $E$'s advantage grows as the actor relies more strongly on critic bootstrapping.

---

## 5. Recommended Stage 1 Search Grids

| Suite | Fixed Shared PPO Params | Stage 1 Swept Grid (Identical per Tuned Algo) | Fixed Aux Params | Stage 1 Tune $\to$ Stage 2 Eval Seeds |
| :--- | :--- | :--- | :--- | :---: |
| **Brax** (9–10 envs) | `GAMMA=0.99`, `GAE_LAMBDA=0.95` (or `0.8`), `VF_CLIP=False`, `VF_COEF` matched | `CRITIC_LR ∈ {3e-4, 1e-3, 3e-3}`<br>`CRITIC_EPOCHS ∈ {4, 8}`<br>`MAX_GRAD_NORM ∈ {1.0, 10.0}`<br>`λ ∈ {0.0, 0.4, 0.8, 0.95}` *(for λ-algos)* | Separate `aux_loss` optimizer, `AUX_VF_COEF=1.0`, `λ_aux=1.0` | `0..2` (3) $\to$ `100..109` (10) |
| **MJX Locomotion** (5 envs) | Playground defaults (`GAMMA=0.97`, `ACTOR_LR=3e-4`, `VF_CLIP=False`, `VF_COEF` matched) | `CRITIC_LR ∈ {3e-4, 1e-3}`<br>`CRITIC_EPOCHS ∈ {4, 8}`<br>`λ ∈ {0.0, 0.6, 0.8, 0.95}` *(for λ-algos)*<br>*(Optional: `MAX_GRAD_NORM ∈ {1.0, 10.0}`)* | Same | `0..2` (3) $\to$ `100..104` (5) |
| **Octax** (4–5 tune $\to$ full eval) | Suite PPO defaults (`VF_CLIP=False`, `VF_COEF` matched) | `CRITIC_LR ∈ {3e-4, 1e-3, 3e-3}`<br>`CRITIC_EPOCHS ∈ {4, 8}`<br>`MAX_GRAD_NORM ∈ {1.0, 10.0}`<br>`λ ∈ {0.0, 0.4, 0.8, 0.95}` *(for λ-algos)* | Same | `0..2` (3) $\to$ `100..109` (10) |
