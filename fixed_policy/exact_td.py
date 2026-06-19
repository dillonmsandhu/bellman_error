# REINFORCE / GRPO-style policy gradient (for intrinsic value)
# uses a timestep dependent variant, based on batch index i.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
import distrax
import core.bellman_error as bellman_error

# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "fixed_td_exact"

def make_train(config):    
    # The saved train state is batched over N_SEEDS (which is 1 by default).
    # We need to extract the parameters for the first seed to remove this extra dimension.
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"]
    
    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n
    
    # Policy to be evaluated
    # model saved under ./results/{alg}/{sub_dir}
    model_dir = 'ppo/' + config['MODEL_LOAD_DIR']
    _, out = utils.load_run_data(model_dir, 'FourRooms-misc', 'results') 
    policy_train_state = out['runner_state'][0]
    def get_policy_matrix():
        "produces pi(.|S) where S is all states"
        policy_params = jax.tree_util.tree_map(lambda x: x[0], policy_train_state.params)
        pi_dist, _ = policy_train_state.apply(policy_params, evaluator.obs_stack)
        pi = pi_dist.probs
        terminal_policy = jnp.ones( [1,n_actions], dtype=pi.dtype) / n_actions
        pi = jnp.vstack([pi, terminal_policy])
        return pi
    
    Pi = get_policy_matrix()
    
    # Get the Markov Chain
    S = evaluator.obs_stack
    P = evaluator.P # 3d tensor S x A x S'
    P_π = jnp.einsum("sa,sam->sm", Pi, P)
    R_π_s = jnp.einsum("sa,sa->s", Pi, evaluator.R)
    # Gymnax awards the reward on the transition *INTO* s'
    R_π = P_π @ R_π_s
    mu = evaluator.compute_stationary_distribution_raw(Pi[:-1, :])
    mu = jnp.append(mu, 0.0)
    
    def train(rng):
        k = config.get('k', 32)
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params,)
                
        def td_loss(params):
            # each update step looks at all observations and produces v_theta(S)            
            v = network.apply(params, S)
            TD_targets = R_π + config['GAMMA'] * P_π @ v
            td_errors = v - jax.lax.stop_gradient(TD_targets)
            loss = 0.5 * jnp.sum(mu * (td_errors ** 2))
            return loss
        
        td_grad = jax.value_and_grad(td_loss)
    
        def td_step(train_state, unused):
            loss, grads = td_grad(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss
        
        # Main Loop
        def _update_step(runner_state, unused):
            train_state, idx = runner_state
            # 1.  Apply expected update NUM_EPOCHS times
            train_state, loss = jax.lax.scan(td_step, train_state, config["NUM_EPOCHS"])
            # 2. Get value metrics and logging
            metric = bellman_error.value_metrics(evaluator, network, train_state.params, random_policy=False)
            metric.update({"total_loss": loss.mean(), "value_loss": loss.mean()})
            
        runner_state = (train_state, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
