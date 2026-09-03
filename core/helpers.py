# helpers.py
# This file contains technical helpers used for the RL loop, including GAE and trace computation, PPO loss, and environment initialization.
from core.imports import *
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper
from envs.log_wrapper import LogWrapper
from envs.wrappers import (NormalizeObservationWrapper, NormalizeRewardWrapper, 
AddChannelWrapper, ClipAction, NormalizeRewardEnvState, NormalizeObsEnvState, 
TerminalInfoWrapper, MountainCarNormalizeWrapper, MountainCarSparseRewardWrapper)
from envs.boyan_chain import MatrixMockEnv, BoyanParams
from envs.whirlpool import WhirlpoolExactValue
from envs.whirlpool_env import Whirlpool
from gymnax.environments import spaces
from flax.core import unfreeze, freeze

def initialize_evaluator(config, env, env_params):
    # for computing the true value
    from envs.fourrooms import FourRoomsExactValue
    from envs.fourrooms_continuing import ContinuingFourRooms
    from envs.boyan_chain import ContinuingBoyanRing
    from envs.mountaincar_exact import MountainCarExactValue
    if not config.get("CALC_TRUE_VALUES", False):
        return None
    
    evaluator = None
    if config['ENV_NAME'] == 'FourRooms-misc':
        evaluator = FourRoomsExactValue(start_pos = env.pos_fixed, goal_pos = env.goal_fixed, fail_prob= env_params.fail_prob,gamma=config['GAMMA']) 
    elif config['ENV_NAME'] == 'FourRooms-cont':
        evaluator = ContinuingFourRooms(start_pos = env.pos_fixed, goal_pos = env.goal_fixed, fail_prob= env_params.fail_prob, gamma=config['GAMMA'])
    elif config['ENV_NAME'] == 'boyan':
        evaluator = ContinuingBoyanRing(gamma=config['GAMMA'], use_visual_obs=True)
    elif config['ENV_NAME'] == 'Whirlpool':
        evaluator = WhirlpoolExactValue(gamma = config['GAMMA'], fail_prob=env_params.fail_prob)
    elif config['ENV_NAME'] == 'MountainCar-v0':
        evaluator = MountainCarExactValue(gamma=config['GAMMA'])
    return evaluator 

def make_env(config):

    if config['ENV_NAME'] == 'MountainCar-v0':
        env, env_params = gymnax.make(config["ENV_NAME"])
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE']
        )
        env = TerminalInfoWrapper(env)
        env = MountainCarNormalizeWrapper(env)
        env = MountainCarSparseRewardWrapper(env)
        config["NETWORK_TYPE"] = 'mlp'

    elif config['ENV_NAME'] == 'FourRooms-misc':
        env, env_params = gymnax.make(config["ENV_NAME"], use_visual_obs=True, goal_fixed=(11,11), pos_fixed = (3,1))
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE'], 
            fail_prob=config['FAIL_PROB']
        )
        env = TerminalInfoWrapper(env)
        
    elif config['ENV_NAME'] == 'FourRooms-cont':
        from envs.wrappers import ContinuingWrapper
        env, env_params = gymnax.make('FourRooms-misc', use_visual_obs=True, goal_fixed=(11,11), pos_fixed = (3,1))
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE'], 
            fail_prob=config['FAIL_PROB']
        )
        env = TerminalInfoWrapper(env)
        env = ContinuingWrapper(env)
        
    elif config['ENV_NAME'] == 'boyan':
        # Create our lightweight mock primitives right here
        env = MatrixMockEnv(size=20, use_visual_obs=config.get("USE_VISUAL_OBS", True))
        env_params = BoyanParams(
            fail_prob=0.0, 
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE']
        )
    elif config['ENV_NAME'] == 'Whirlpool':
        from envs.whirlpool_env import EnvParams
        env = Whirlpool(size = 13, use_visual_obs=True)
        env_params = EnvParams(fail_prob = config['FAIL_PROB'])
        env = TerminalInfoWrapper(env)
        # We skip TerminalInfoWrapper/ContinuingWrapper since it's a pure matrix evaluator,
        # but it will safely pick up the downstream wrappers (LogWrapper, etc.) via its properties!

    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
    
    print('Env:', config['ENV_NAME'])
    print('Default Obs Shape:', env.observation_space(env_params).shape)
    
    env = LogWrapper(env)
    
    if isinstance(env.action_space(env_params), spaces.Box):
        env = ClipAction(env)
    
    if config["NETWORK_TYPE"] == "mlp":
        if len(env.observation_space(env_params).shape) > 1:
            env = FlattenObservationWrapper(env)
    if config["NETWORK_TYPE"] == "cnn":
        if len(env.observation_space(env_params).shape) < 3:
            env = AddChannelWrapper(env)
    if config["NORMALIZE_OBS"]:
        env = NormalizeObservationWrapper(env) 
    
    print('Obs Shape:', env.observation_space(env_params).shape)
    print('Action Shape:', env.action_space(env_params).shape)
    return env, env_params
    
def _loss_fn(params, network, traj_batch, gae, targets, config):
    # Critic loss
    value_loss = v_loss_fn(params, network, traj_batch, gae, targets, config)

    # Actor loss
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        config['POLICY_COEFF'] * loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)

def _loss_fn_no_w(params, network, traj_batch, gae, targets, config):
    # Critic loss
    value_loss = no_w_v_loss_fn(params, network, traj_batch, gae, targets, config)

    # Actor loss
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)    

def pi_loss_fn(params, network, traj_batch, gae, config):
    pi = network.apply(params, traj_batch.obs, method=network.policy)
    log_prob = pi.log_prob(traj_batch.action)

    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    A_CLIP = config.get('ADV_CLIP', 3.0)
    gae = jnp.clip(gae, -A_CLIP, A_CLIP) # outlier clipping for the policy. 95% unclipped with 2.
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - config["CLIP_EPS"],
            1.0 + config["CLIP_EPS"],
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    return loss_actor, entropy

def ppo_clipped_v_loss(traj_batch, value_pred, targets, config):
    e = config["VF_CLIP"]
    value_pred_clipped = traj_batch.value + (
        value_pred - traj_batch.value).clip(-e,e)
    value_losses = jnp.square(value_pred - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    
def v_loss_fn(params, network, traj_batch, gae, targets, config):
    # VALUE LOSS
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_loss = ppo_clipped_v_loss(traj_batch, value_pred, targets, config)
    total_loss = config["VF_COEF"] * value_loss
    return total_loss

def v_loss_fn_laplacian_smoothing(params, network, traj_batch, targets, config):
    gamma = config["GAMMA"]
    c = config["VF_CLIP"]
    # 1. Current State Predictions & Errors (e_i)
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_pred_clipped = traj_batch.value + (
        value_pred - traj_batch.value).clip(-c,c)
    e_i = targets - value_pred_clipped 
    base_ve_loss = 0.5 * jnp.mean(e_i ** 2)
    # 2. Next State Predictions & Errors (e_j)
    # Requires traj_batch to contain the adjacent (s, G) pairs
    next_value_pred = network.apply(params, traj_batch.next_obs, method=network.value)
    next_v_fixed = jax.lax.stop_gradient(next_value_pred)
    next_value_pred_clipped = next_v_fixed + (
        next_value_pred - next_v_fixed).clip(-c,c)
    e_j = traj_batch.next_target - next_value_pred_clipped
    valid_mask = 1.0 - traj_batch.done
    n_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)
    laplacian_loss = 0.5 * jnp.sum(valid_mask * (e_i - e_j) ** 2) / n_valid
    # Combine using the exact Dirichlet expansion weights
    weight_laplacian = gamma * config['LAPLACE_SMOOTHING_COEFF']
    dirichlet_value_loss = (1 - weight_laplacian) * base_ve_loss + weight_laplacian * laplacian_loss
    total_loss = config["VF_COEF"] * dirichlet_value_loss
    return total_loss, {"base_ve_loss": base_ve_loss, "laplacian_loss": laplacian_loss, "total_loss": total_loss}

def no_w_v_loss_fn(params, network, traj_batch, gae, targets, config):
    # ---------------------------------------------------------
    # Firewalled Parameters
    # ---------------------------------------------------------
    def freeze_w_map(path, val):
        is_w = any(getattr(p, 'key', None) in ('w_layer', 'critic_head') or 
                   'w_layer' in str(p) or 'critic_head' in str(p) for p in path)
        return jax.lax.stop_gradient(val) if is_w else val
    
    params_w_frozen = jax.tree_util.tree_map_with_path(freeze_w_map, params)
    value_for_phi = network.apply(params_w_frozen, traj_batch.obs, method = network.value)
    loss_phi = ppo_clipped_v_loss(traj_batch, value_for_phi, targets, config)
    return config["VF_COEF"] * loss_phi

def v_loss_fn_no_grad(params, network, traj_batch, gae, targets, config):
    "No update to phi."
    # 1. Forward pass through the CNN to get the features
    phi = network.apply(params, traj_batch.obs, method=network.value_features)
    
    # 2. SEVER THE GRAPH: Gradients from the value loss cannot pass this point.
    # The CNN weights will receive zero gradient from this loss function.
    phi_freeze = jax.lax.stop_gradient(phi)
    
    # 3. Forward pass through ONLY the linear head using the frozen features
    value_pred = network.apply(params, phi_freeze, method=network.value_from_features)
    value_loss = ppo_clipped_v_loss(traj_batch, value_pred, targets, config)
    total_loss = config["VF_COEF"] * value_loss
    return total_loss

def shuffle_and_batch(rng, transitions, n_minibatches):
    def preprocess_transition(x, rng):
        x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs (batch_size), ...
        x = jax.random.permutation(rng, x)  # shuffle the transitions
        x = x.reshape(n_minibatches, -1, *x.shape[1:])  # num_mini_updates, batch_size/num_mini_updates, ...
        return x
    minibatches = jax.tree.map(lambda x: preprocess_transition(x, rng), transitions)  # num_actors*num_envs (batch_size), ...
    return minibatches


def add_values_to_metric(config, metric, evaluator, network, train_state, traj_batch, compute_true_vals = True):
    """Uses evaluator to compute the per-state quantities and append them to metric."""
    if evaluator:
        pi, v_pred = network.apply(train_state.params, evaluator.obs_stack)
        pi = jnp.vstack([pi, jnp.zeros((1, pi.shape[-1]))]) # assumes terminal state.
        
        Φ = network.apply(train_state.params, evaluator.obs_stack, method=network.value_features)
        Φ = jnp.vstack([Φ, jnp.zeros((1, Φ.shape[-1]))])  # assumes terminal state.
        
        v_pred = network.apply(train_state.params, Φ, method=network.value_from_features)
    
    # True value
    if compute_true_vals:
        # The evaluator dictates the exact ground truth shapes here
        v = evaluator.compute_true_values(pi)

    # 4. Visitation Logic
    obs = jnp.asarray(traj_batch.obs)
    next_obs = jnp.asarray(traj_batch.next_obs)
    env_name = config.get("ENV_NAME", "")
    
    if env_name in {"FourRooms-misc", "FourRoomsCustom-v0"} or "SparseMaze" in env_name:
        if obs.ndim >= 5:
            metric['visitation_count'] = next_obs[..., 1].sum(axis=(0, 1))
        elif obs.ndim >= 3 and obs.shape[-1] >= 2:
            size = traj_batch.reward.shape[0] 
            pos = next_obs[..., :2].astype(jnp.int32)
            y = jnp.clip(pos[..., 0], 0, size - 1).reshape(-1)
            x = jnp.clip(pos[..., 1], 0, size - 1).reshape(-1)
            counts = jnp.zeros((size, size), dtype=jnp.float32)
            metric['visitation_count'] = counts.at[y, x].add(1.0)
    
    # 5. Error Metrics (Perfect shape alignment guaranteed by the evaluator)    
    metric.update({
        "v": v,
        "v_pred": v_pred,
        "pi": pi,
        "Empirical MSVE": jnp.mean((v - v_pred)**2) ,
    })
    
    return metric


# def calculate_gae(traj_batch, γ, λ,):

#     def _get_advantages(gae, transition):
#         done = transition.done

#         delta = transition.reward + γ * transition.next_value * (1 - done) - transition.value
#         gae = delta + (γ * λ * (1 - done) * gae)
        
#         return gae, gae

#     initial_accs = jnp.zeros_like(traj_batch.value[0])
#     _, advantages = jax.lax.scan(
#         _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
#     )
    
#     return (advantages, advantages + traj_batch.value)

def calculate_gae(traj_batch, γ, λ):
    def _get_advantages(gae, transition):
        done = transition.done
        is_timeout = transition.info["is_timeout"]

        # MASK 1: Value Bootstrapping
        true_terminal = done & ~is_timeout
        bootstrap_mask = 1.0 - true_terminal

        # MASK 2: GAE Accumulation (Trajectory Boundary)
        # Sever the GAE chain if the environment reset for ANY reason (terminal or timeout).
        # The 'gae' variable coming from the future belongs to a different episode.
        boundary_mask = 1.0 - done

        # 1. Compute TD Error (Safely bootstraps through timeouts)
        delta = transition.reward + γ * transition.next_value * bootstrap_mask - transition.value
        
        # 2. Accumulate GAE (Safely breaks at episode resets)
        gae = delta + (γ * λ * boundary_mask * gae)
        
        return gae, gae

    initial_accs = jnp.zeros_like(traj_batch.value[0])
    _, advantages = jax.lax.scan(
        _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
    )
    
    return (advantages, advantages + traj_batch.value)


def find_closest_divisor(total, requested):
    for n in range(requested, 0, -1):
        if total % n == 0:
            return n
    return 1

def inject_weights(train_state, w):
    """Overwrites the critic_head weights, preserving the original PyTree type."""
    # 1. Slice the weights (last dim is bias)
    kernel_weights = jnp.expand_dims(w[:-1], axis=-1)
    bias_weight = w[-1:]
    
    # 2. Define the new layer dictionary
    new_head = {
        'kernel': kernel_weights,
        'bias': bias_weight
    }
    
    # 3. Inject it while preserving the container type (dict vs frozendict)
    params = train_state.params
    if not isinstance(params, dict):
        params = unfreeze(params)
    
    new_params = dict(params)
    new_params['params'] = dict(new_params['params'])
    if 'critic_head' in new_params['params']:
        new_params['params']['critic_head'] = new_head
    if 'w_layer' in new_params['params']:
        new_params['params']['w_layer'] = new_head

    if isinstance(train_state.params, dict):
        return train_state.replace(params=new_params)
    else:
        return train_state.replace(params=freeze(new_params))

def get_evaluation_policies(base_config, evaluator):
    if base_config.get("USE_GREEDY_POLICY", False):
        import core.bellman_error as bellman_error
        import distrax
        if hasattr(evaluator, "get_optimal_value_function"):
            V_star = evaluator.get_optimal_value_function()
        else:
            V_star = jnp.zeros(evaluator.num_total_states)
        
        greedy_actions = bellman_error.compute_greedy_policy(evaluator.P, evaluator.R, evaluator.gamma, V_star)
        pi_greedy = jax.nn.one_hot(greedy_actions, evaluator.num_actions)
        epsilon = base_config.get("POLICY_EPSILON", 0.0)
        pi_eps = (1 - epsilon) * pi_greedy + (epsilon / evaluator.num_actions) * jnp.ones_like(pi_greedy)
        
        def policy_fn(obs):
            obs_flat = obs.reshape((obs.shape[0], -1)) if obs.ndim > 1 else obs.flatten()[None, :]
            stack_flat = evaluator.obs_stack.reshape((evaluator.obs_stack.shape[0], -1))
            diffs = jnp.sum((obs_flat[:, None, :] - stack_flat[None, :, :])**2, axis=-1)
            state_indices = jnp.argmin(diffs, axis=-1)
            probs = pi_eps[state_indices]
            return distrax.Categorical(probs=probs)
            
        terminal_policy = jnp.ones([1, evaluator.num_actions], dtype=pi_eps.dtype) / evaluator.num_actions
        policy_matrix = jnp.vstack([pi_eps, terminal_policy])
        return policy_fn, policy_matrix
    else:
        import core.utils as utils
        model_dir = 'ppo/' + base_config['MODEL_LOAD_DIR']
        _, out = utils.load_run_data(model_dir, base_config['ENV_NAME'], 'results') 
        policy_train_state = out['runner_state'][0]
        policy_params = jax.tree_util.tree_map(lambda x: x[0], policy_train_state.params)
        
        def policy_fn(obs):
            pi, _ = policy_train_state.apply_fn(policy_params, obs)
            # handle cases where apply_fn returns a tuple (pi, value) or just pi
            if isinstance(pi, tuple):
                pi = pi[0]
            return pi
            
        # build the matrix
        pi_dist = policy_fn(evaluator.obs_stack)
        pi_probs = pi_dist.probs
        terminal_policy = jnp.ones([1, evaluator.num_actions], dtype=pi_probs.dtype) / evaluator.num_actions
        policy_matrix = jnp.vstack([pi_probs, terminal_policy])
        
        return policy_fn, policy_matrix
