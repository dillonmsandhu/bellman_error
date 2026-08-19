# dispatcher for the CLI, redirects to either evaluate.py or sweep.py.
from evaluate import evaluate
from sweep import tune
import jax

def run_experiment_main(make_train, SAVE_DIR):
    import argparse
    import datetime
    import traceback
    import core.helpers as helpers
    import core.config as config
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run-suffix', type=str, default=run_timestamp)
    parser.add_argument('--n-seeds', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--save-metrics', action='store_true')
    parser.add_argument('--env-ids', nargs='+', default=[])

    args = parser.parse_args()

    config = config.config
    env_list = [config.get('ENV_NAME')]

    # 3. Environment Priority (CLI takes precedence)
    if args.env_ids:
        env_list = args.env_ids

    for i, env_name in enumerate(env_list):
        if env_name is None: continue
            
        # Create a clean copy for this specific environment run
        run_config = config.copy()
        run_config['ENV_NAME'] = env_name
        
        # Apply command-line JSON overrides if they exist
        if args.config:
            from core.utils import parse_config_override
            run_config.update(parse_config_override(args.config))
            
        if args.n_seeds > 0:
            run_config['N_SEEDS'] = args.n_seeds

        print(f"\n{'='*50}")
        print(f"RUNNING ENV {i+1}/{len(env_list)}: {env_name}")
        print(f"Config: {args.config}")
        # print(f"Network: {run_config.get('NETWORK_TYPE')}")
        print(f"{'='*50}")
        
        rng = jax.random.PRNGKey(run_config.get('SEED', 42))
        
        try:
            # Note: make_train and evaluate should be defined in your scope
            evaluate(run_config, make_train, SAVE_DIR, args, rng)
        except Exception as e:
            print(f"!!! CRITICAL ERROR running {env_name} !!!")
            traceback.print_exc()
            print("Continuing to next environment...")
