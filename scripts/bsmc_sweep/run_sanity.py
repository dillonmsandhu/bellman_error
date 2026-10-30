"""
scripts/bsmc_sweep/run_sanity.py

Fast local sanity check verifying:
  1. All 5 algorithms (td_lambda, e_lambda, e_lambda_fwd, bsmc_aux, bsmc_fwd)
  2. JAX JIT compilation and shapes
  3. Analytical MC weight calculation
  4. Metric collection and CSV output
Runs on FourRooms-misc for 2 seeds, 3 updates.
"""

import os
import sys
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def main():
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts/bsmc_sweep/run_single_env.py"),
        "--env_name", "FourRooms-misc",
        "--lambdas", "0.8", "0.95",
        "--num_updates", "3",
        "--n_seeds", "2",
        "--critic_lrs", "0.001",
        "--num_envs", "16",
        "--num_steps", "64",
        "--minibatch_size", "64",
        "--num_epochs", "2",
        "--out_dir", "scratch/bsmc_sanity_test",
        "--algos", "td_lambda", "e_lambda", "e_lambda_fwd", "bsmc_aux", "bsmc_fwd",
    ]
    print(f"Running command:\n{' '.join(cmd)}\n")
    subprocess.check_call(cmd, cwd=REPO_ROOT)
    print("\nSanity test passed successfully!")

if __name__ == "__main__":
    main()
