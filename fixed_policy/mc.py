# Sample-based Monte Carlo on fixed policy (TD with GAE_LAMBDA=1.0)
import core.utils as utils
from fixed_policy.td import make_train as make_td_train

SAVE_DIR = "fixed_mc"

def make_train(base_config):
    config = base_config.copy()
    config["GAE_LAMBDA"] = 1.0
    config["VALUE_LAMBDA"] = 1.0
    return make_td_train(config)

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
