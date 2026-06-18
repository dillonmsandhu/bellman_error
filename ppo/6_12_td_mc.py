# Does TD(0) on the final weights and TD(1) to learn the feature extractor.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.bellman_error as bellman_error
# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "ppo_td0"

def td0_w_loss_fn(params, network, traj_batch, config):
    # 1. Extract and freeze current state features
    phi = network.apply(params, traj_batch.obs, method=network.value_features)
    phi = jax.lax.stop_gradient(phi)
    
    # 2. Extract and freeze next state features 
    # (Using real_next_obs to correctly handle environment boundaries)
    next_phi = network.apply(params, traj_batch.info['real_next_obs'], method=network.value_features)
    next_phi = jax.lax.stop_gradient(next_phi)
    
    # 3. Compute Value predictions strictly from the linear head
    v_pred = network.apply(params, phi, method=network.value_from_features)
    next_v_pred = network.apply(params, next_phi, method=network.value_from_features)
    
    # 4. Construct Semi-Gradient TD(0) Target
    # Target is detached from the computational graph
    target = traj_batch.reward + config["GAMMA"] * (1 - traj_batch.done) * next_v_pred
    target = jax.lax.stop_gradient(target)
    
    # 5. Standard MSE Loss for the linear projection
    # (PPO clipping is generally omitted here so the linear system can freely 
    # converge to the TD fixed point).
    loss = 0.5 * jnp.mean(jnp.square(v_pred - target))
    
    return loss

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    obs_shape = env.observation_space(env_params).shape

    def train(rng):
        k = config.get('k', 32)
        network, network_params = networks.initialize_network(rng, obs_shape, env, env_params, k, n_heads=2, layer_norm=config['LAYER_NORM'])
        train_state = networks.initialize_flax_train_state_no_w(config, network, network_params,)
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info['real_next_obs']
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)

                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])


            # --- ADVANTAGE CALCULATION ---
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], config["POLICY_LAMBDA"])
            _, target = helpers.calculate_gae(traj_batch, config["GAMMA"], 1.0)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    
                    # ---------------------------------------------------------
                    # 1. MC / PPO Gradients (Trains Actor & Features, not w)
                    # ---------------------------------------------------------
                    # _loss_fn_no_w already stops gradients to 'w'.
                    grad_mc_fn = jax.value_and_grad(helpers._loss_fn_no_w, has_aux=True)
                    (mc_total, (v_loss, a_loss, ent)), grads_mc = grad_mc_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    # ---------------------------------------------------------
                    # 2. TD(0) Gradients (Trains Final Layer w ONLY)
                    # ---------------------------------------------------------
                    grad_td_fn = jax.value_and_grad(td0_w_loss_fn)
                    td_loss, grads_td = grad_td_fn(train_state.params, network, traj_batch, config)
                    # ---------------------------------------------------------
                    # 3. Combine and Apply
                    # ---------------------------------------------------------
                    td0_coef = config.get("TD0_W_COEF", 1.0)
                    
                    # Merge the gradient trees. Active layers get their respective 
                    # gradients; frozen layers get 0.0 + active_grad.
                    combined_grads = jax.tree_util.tree_map(
                        lambda g_mc, g_td: g_mc + (td0_coef * g_td), 
                        grads_mc, 
                        grads_td
                    )
                    
                    train_state = train_state.apply_gradients(grads=combined_grads)
                    
                    return train_state, (mc_total, v_loss, a_loss, ent, td_loss)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                
                # shuffle_and_batch handles the flattening of [NUM_STEPS, NUM_ENVS]
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                train_state, loss_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), loss_info

            initial_update_state = (train_state, traj_batch, advantages, target, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state
            
            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            # Shared Metrics
            metric.update(
                {
                    "total_loss": loss_info[0].mean(),
                    "value_loss": loss_info[1].mean(),
                    "actor_loss": loss_info[2].mean(),
                    "entropy": loss_info[3].mean(),
                    "w_td0_loss": loss_info[4].mean(),  # New TD(0) loss logged here
                    "mean_rew": traj_batch.reward.mean(),
                }
            )
            
            value_metrics = bellman_error.value_metrics(evaluator, network, train_state.params, random_policy=False)
            metric.update(value_metrics)

            runner_state = (train_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)