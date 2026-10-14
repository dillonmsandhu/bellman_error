"""
ppo/oct5.py

PPO with Oct 5 (Corrected) Sampled E Critic:
Critic Loss = Sept 14 Loss + 0.5 * gamma * (mean(e_i^2) - mean(e_j^2))
Actor Loss = Standard PPO clipped surrogate objective with GAE advantages.
"""

from ppo.sampled_E import make_train as make_base_train

SAVE_DIR = "ppo/oct5"

def make_train(base_config):
    cfg = base_config.copy()
    cfg["E_VARIANT"] = "corrected"
    return make_base_train(cfg)

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
