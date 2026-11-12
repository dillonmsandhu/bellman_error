# Project Context: `bellman_error` & `E_min`

## Repository Roles & Workflow
This repository (`bellman_error`) is part of a dual-repository research project alongside `E_min`:

1. **`bellman_error` (R&D, Diagnostics & Prototyping)**:
   - Primary purpose: Getting algorithms to work, theoretical validation, closed-form matrix calculations, and deep diagnostics on tabular / exact environments.
   - **Key Breakthrough**: Holds the validated, strong formulation of the **E-minimization algorithm using `v_aux`** (auxiliary value network / debiased sampled E-minimization). Key locations include:
     - `random_policy/E_sample_v_aux_surgical.py`, `random_policy/E_sample_v_aux_full.py`, `random_policy/E_sample_v_aux_bilinear.py`
     - `ppo/sampled_E.py`
     - `core/sampling_e_metrics.py`, `core/sampling_e_lambda_metrics.py`
     - `scratch/test_v_aux_bootstrapping.py`
     - `reference/ai/ai/aux_sampled_e_lambda.md`, `reference/ai/ai/delta_aux_e_min.md`
   - Sibling symlink: `external_E_min` -> `/Users/dillonsandhu/Documents/Research/E_min`

2. **`E_min` (Final Simulators & Scaling)**:
   - Path: `/Users/dillonsandhu/Documents/Research/E_min`
   - Primary purpose: The likely final publication-facing repository containing pure JAX implementations for:
     - **Gymnax** (vectorized discrete & continuous RL environments)
     - **Brax** (upcoming continuous physics environments)
     - Slurm GPU cluster sweeps
   - Algorithms prototyped and debugged in `bellman_error` (especially `v_aux` E-min) are ported into `E_min`.
   - Sibling symlink: `external_bellman_error` -> `/Users/dillonsandhu/Documents/Research/bellman_error`

## Agent Guidelines
- Any agent operating in this repository has full authority to inspect, read, compare, and modify code in `E_min`.
- Follow the fail-fast invariants documented in `.agents/rules/non_defensive_coding.md` across both repositories.
- When validating algorithms in `bellman_error` and porting to `E_min`, always run a local CPU smoke test in `E_min` before Slurm submissions.
