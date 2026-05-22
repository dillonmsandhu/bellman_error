# helpers.py
# This file contains technical helpers used for the RL loop, including GAE and trace computation, PPO loss, and environment initialization.
from core.imports import *
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper
from envs.log_wrapper import LogWrapper
from envs.wrappers import NormalizeObservationWrapper, NormalizeRewardWrapper, AddChannelWrapper, ClipAction, NormalizeRewardEnvState, NormalizeObsEnvState, TerminalInfoWrapper
from gymnax.environments import spaces
from flax.core import unfreeze, freeze

def initialize_evaluator(config, env, env_params):
    from envs.fourrooms import FourRoomsExactValue
    
    if not config.get("CALC_TRUE_VALUES", False):
        return None
    
    evaluator = None
    if config['ENV_NAME'] == 'FourRooms-misc':
        evaluator = FourRoomsExactValue(start_pos = env.pos_fixed, goal_pos = env.goal_fixed, fail_prob= env_params.fail_prob) # for computing the true 
    
    return evaluator 


def make_env(config):

    if config['ENV_NAME'] == 'FourRooms-misc':
        env, env_params = gymnax.make(config["ENV_NAME"], use_visual_obs=True, goal_fixed=(11,11), pos_fixed = (3,1))
        env_params = env_params.replace(
            max_steps_in_episode=config['MAX_STEPS_IN_EPISODE'], 
            fail_prob=config['FAIL_PROB']
        )

    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
    
    print('Env:', config['ENV_NAME'])
    print('Default Obs Shape:', env.observation_space(env_params).shape)
    
    env = TerminalInfoWrapper(env) # adds the terminal state to info. also adds goal information.
    env = LogWrapper(env)
    
    if isinstance(env.action_space(env_params), spaces.Box):
        env = ClipAction(env) # Ensures sampled actions are within [low, high]
    
    if config["NETWORK_TYPE"] == "mlp":
        env = FlattenObservationWrapper(env)
    if config["NETWORK_TYPE"] == "cnn":
        if len(env.observation_space(env_params).shape) < 3:
            env = AddChannelWrapper(env) # add an empty channel to the end if 2d input
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
        loss_actor
        + config["VF_COEF"] * value_loss
        - config["ENT_COEF"] * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)

def _loss_fn_cd(params, network, traj_batch, gae, targets, config):
    
    # Critic loss
    value_loss = coordinate_descent_v_loss_fn(params, network, traj_batch, gae, targets, config)

    # Actor loss
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        loss_actor
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

def v_loss_fn(params, network, traj_batch, gae, targets, config):
    # VALUE LOSS
    value = network.apply(params, traj_batch.obs, method=network.value)
    value_pred_clipped = traj_batch.value + (
        value - traj_batch.value
    ).clip(-config["VF_CLIP"], config["VF_CLIP"])
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )

    total_loss = config["VF_COEF"] * value_loss
    return total_loss

def coordinate_descent_v_loss_fn(params, network, traj_batch, gae, targets, config):
    # ---------------------------------------------------------
    # 1. Create the Firewalled Parameters
    # ---------------------------------------------------------
    def freeze_w_map(path, val):
        is_w = any('critic_head' in str(p) for p in path)
        return jax.lax.stop_gradient(val) if is_w else val

    def freeze_phi_map(path, val):
        is_phi = any('critic_cnn' in str(p) for p in path)
        return jax.lax.stop_gradient(val) if is_phi else val

    params_w_frozen = jax.tree_util.tree_map_with_path(freeze_w_map, params)
    params_phi_frozen = jax.tree_util.tree_map_with_path(freeze_phi_map, params)

    # ---------------------------------------------------------
    # 2. PPO Clipped Loss Helper
    # ---------------------------------------------------------
    def ppo_clipped_loss(value_pred):
        value_pred_clipped = traj_batch.value + (value_pred - traj_batch.value).clip(
            -config["VF_CLIP"], config["VF_CLIP"]
        )
        value_losses = jnp.square(value_pred - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    # ---------------------------------------------------------
    # 3. Decoupled Forward Passes & Losses
    # ---------------------------------------------------------
    # Train Phi: Forward pass with frozen W
    value_for_phi = network.apply(params_w_frozen, traj_batch.obs, method = network.value)
    loss_phi = ppo_clipped_loss(value_for_phi)

    # Train W: Forward pass with frozen Phi
    value_for_w = network.apply(params_phi_frozen, traj_batch.obs, method = network.value)
    loss_w = ppo_clipped_loss(value_for_w)

    # ---------------------------------------------------------
    # 4. Combine
    # ---------------------------------------------------------
    
    total_loss = config["VF_COEF"] * (loss_phi + loss_w)
    
    return total_loss

def no_w_v_loss_fn(params, network, traj_batch, gae, targets, config):
    # ---------------------------------------------------------
    # 1. Create the Firewalled Parameters
    # ---------------------------------------------------------
    def freeze_w_map(path, val):
        is_w = any('critic_head' in str(p) for p in path)
        return jax.lax.stop_gradient(val) if is_w else val

    params_w_frozen = jax.tree_util.tree_map_with_path(freeze_w_map, params)

    # ---------------------------------------------------------
    # 2. PPO Clipped Loss Helper
    # ---------------------------------------------------------
    def ppo_clipped_loss(value_pred):
        value_pred_clipped = traj_batch.value + (value_pred - traj_batch.value).clip(
            -config["VF_CLIP"], config["VF_CLIP"]
        )
        value_losses = jnp.square(value_pred - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    # Train Phi: Forward pass with frozen W
    value_for_phi = network.apply(params_w_frozen, traj_batch.obs, method = network.value)
    loss_phi = ppo_clipped_loss(value_for_phi)
    return config["VF_COEF"] * loss_phi


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
    new_critic_head = {
        'kernel': kernel_weights,
        'bias': bias_weight
    }
    
    # 3. Inject it while preserving the container type (dict vs frozendict)
    if isinstance(train_state.params, dict):
        # Newer Flax (Standard Dict): Use a functional copy to avoid mutating JAX Tracers in-place
        new_params = dict(train_state.params)
        new_params['params'] = dict(new_params['params'])
        new_params['params']['critic_head'] = new_critic_head
        return train_state.replace(params=new_params)
        
    else:
        # Older Flax (FrozenDict): Safe to use unfreeze/freeze
        params = unfreeze(train_state.params)
        params['params']['critic_head'] = new_critic_head
        return train_state.replace(params=freeze(params))
