"""
ppo/oct8.py

PPO with Oct 8 (Start-Anchored) Sampled E Critic:
Critic Loss = Sept 14 Loss + 0.5 * gamma * nu_hat * mean(e_start^2)
where nu_hat is an Adam-style bias-corrected running average of terminal rates.
Actor Loss = Standard PPO clipped surrogate objective with GAE advantages.
"""

from ppo.sampled_E import make_train as make_base_train

SAVE_DIR = "ppo/oct8"

def make_train(base_config):
    cfg = base_config.copy()
    cfg["E_VARIANT"] = "oct8"
    return make_base_train(cfg)

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
