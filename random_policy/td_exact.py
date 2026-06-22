# REINFORCE / GRPO-style policy gradient (for intrinsic value)
# uses a timestep dependent variant, based on batch index i.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
from flax.training.train_state import TrainState
import core.bellman_error as bellman_error

# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "random_td_exact"

def make_train(config):    
    # The saved train state is batched over N_SEEDS (which is 1 by default).
    # We need to extract the parameters for the first seed to remove this extra dimension.
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"]
    config['NUM_ENVS'] = 1
    config['NUM_STEPS'] = 1
    config['NUM_EPOCHS'] = 1
    
    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n
    
    # Policy to be evaluated
    def get_random_policy_matrix(num_states: int, num_actions: int) -> jax.Array:
        """
        Produces a uniform random policy matrix PI of shape (num_total_states, num_actions).
        
        Args:
            num_states: The number of active states in the environment.
            num_actions: The total number of available actions.
        """
        # 1. Create uniform distribution for active states (1/N probability per action)
        pi_active = jnp.ones((num_states, num_actions)) / num_actions
        
        # 2. Create uniform distribution for the single terminal state
        pi_terminal = jnp.ones((1, num_actions)) / num_actions
        
        # 3. Stack them to match your evaluator's S+1 state requirement
        pi = jnp.vstack([pi_active, pi_terminal])
        
        return pi
    
    Pi = get_random_policy_matrix(evaluator.num_states, evaluator.num_actions)
    
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
        # Initialize Network
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        total_grad_steps = config["NUM_UPDATES"] * config["NUM_EPOCHS"]
        lr_scheduler = optax.linear_schedule(config["LR"], config["LR_END"], total_grad_steps)
        tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adamw(lr_scheduler, 
                weight_decay = config.get('WEIGHT_DECAY', 1e-2),
                eps=config.get('ADAM_EPS', 1e-5)
                ),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        runner_state = (train_state, 1)
        
        def td_loss(params):
            # each update step looks at all observations and produces v_theta(S)            
            print(S.shape)
            v = network.apply(params, S) # 104 states, no terminal
            v = jnp.append(v, 0.0)
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
            train_state, loss = jax.lax.scan(td_step, train_state, None, config["NUM_EPOCHS"])
            # 2. Get value metrics and logging
            metric = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=False, target_policy_fn=get_policy
            )
            metric.update({"total_loss": loss.mean(), "value_loss": loss.mean()})
            runner_state = (train_state, idx + 1)
            return runner_state, metric
            
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
