# Sample-based Monte Carlo PPO (TD with VALUE_LAMBDA=1.0)
import core.utils as utils
from ppo.sampled_td_lambda import make_train as make_td_lambda_train

SAVE_DIR = "ppo/mc"

def make_train(base_config):
    config = base_config.copy()
    config["VALUE_LAMBDA"] = 1.0
    train_fn = make_td_lambda_train(config)

    def train(rng, hparams=None):
        if hparams is not None:
            hparams = dict(hparams)
            hparams["VALUE_LAMBDA"] = 1.0
        return train_fn(rng, hparams)

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
