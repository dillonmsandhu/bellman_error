# Sept 14 (Uncorrected) Sampled E-loss for Random Policy Evaluation
# L = (1-gamma) E[e_i^2] + (gamma/2) E[(e_i - e_j)^2]
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import distrax
import core.bellman_error as bellman_error
from core.feature_metrics import feature_metrics
import core.utils as utils
from core.gradient_tracking import extract_critic_flat_grads, cos_similarity

SAVE_DIR = "random/sep14"

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    next_target: jnp.ndarray
    is_start: jnp.ndarray
    info: jnp.ndarray

def make_train(base_config):
    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    minibatch_size = base_config.get("MINIBATCH_SIZE", 1024)
    base_config["NUM_MINIBATCHES"] = max(1, batch_size // minibatch_size)
    base_config["NUM_UPDATES"] = max(1, base_config["TOTAL_TIMESTEPS"] // batch_size)
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    is_continuous = isinstance(env.action_space(env_params), spaces.Box)
    if is_continuous:
        action_shape = env.action_space(env_params).shape
        n_actions = getattr(evaluator, "num_actions", action_shape[0])
    else:
        action_shape = ()
        n_actions = env.action_space(env_params).n

    # --- Precompute Exact S Matrix and V_true for Random Policy ---
    n_states = evaluator.num_states
    pi_active = jnp.ones((n_states, n_actions)) / n_actions
    terminal_policy = jnp.ones((1, n_actions)) / n_actions
    pi_full = jnp.vstack([pi_active, terminal_policy])

    V_true = evaluator.compute_true_values_raw(pi_full)
    mu = evaluator.compute_stationary_distribution_raw(pi_active)[0]
    mu_full = jnp.append(mu, 0.0)
    D = jnp.diag(mu_full)
    P_pi = jnp.einsum("sa,sam->sm", pi_full, evaluator.P)
    gamma = base_config["GAMMA"]
    A_mat = D @ (jnp.eye(evaluator.num_total_states) - gamma * P_pi)
    S_mat = 0.5 * (A_mat + A_mat.T)

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config['k']
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        last_true_terminal = jnp.zeros(config["NUM_ENVS"], dtype=bool)

        def exact_e_loss(params):
            v = network.apply(params, evaluator.obs_stack, method=network.value).squeeze()
            diff = V_true - jnp.append(v, 0.0)
            return diff.T @ S_mat @ diff

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, rng, idx = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                value = network.apply(train_state.params, last_obs)
                if is_continuous:
                    pi = distrax.Independent(
                        distrax.Uniform(
                            low=jnp.full((config['NUM_ENVS'], *action_shape), -1.0),
                            high=jnp.full((config['NUM_ENVS'], *action_shape), 1.0),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                else:
                    pi = distrax.Categorical(logits=jnp.zeros((config['NUM_ENVS'], n_actions)))
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info['real_next_obs']
                next_val = network.apply(train_state.params, true_next_obs)

                if "is_timeout" in info:
                    is_timeout = info["is_timeout"]
                else:
                    is_timeout = jnp.zeros_like(done, dtype=bool)
                true_terminal = done & ~is_timeout
                is_start = last_true_terminal

                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, 0.0, is_start, info
                )
                return (train_state, env_state, obsv, true_terminal, rng), transition

            env_step_state = (train_state, env_state, last_obs, last_true_terminal, rng)
            (_, env_state, last_obs, last_true_terminal, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # MC Return calculation (GAE lambda=1.0)
            return_lambda = config.get("RETURN_LAMBDA", 1.0)
            advantages, targets = helpers.calculate_gae(traj_batch, config["GAMMA"], return_lambda)
            
            is_timeout = traj_batch.info["is_timeout"]
            true_terminal = traj_batch.done & ~is_timeout

            # Align next targets G_{t+1}
            next_targets = jnp.roll(targets, shift=-1, axis=0)
            next_targets = next_targets.at[-1].set(traj_batch.next_value[-1])
            next_targets = jnp.where(is_timeout, traj_batch.next_value, next_targets)
            next_targets = jnp.where(true_terminal, 0.0, next_targets)
            traj_batch = traj_batch._replace(next_target=next_targets)

            # Sept 14 (Uncorrected) Loss Function
            def E_loss_fn(params, traj_batch, targets):
                gamma = config["GAMMA"]
                v_i = network.apply(params, traj_batch.obs)
                v_j = network.apply(params, traj_batch.next_obs)
                true_term = traj_batch.done & ~traj_batch.info["is_timeout"]
                
                value_loss, magnitude_loss, laplacian_loss = helpers.e_critic_loss(
                    v_i, targets, v_j, traj_batch.next_target, true_term, gamma, variant="sep14"
                )
                
                total_loss = config["VF_COEF"] * value_loss
                return total_loss, {
                    "total_loss": total_loss,
                    "magnitude_loss": magnitude_loss,
                    "laplacian_loss": laplacian_loss,
                    "value_loss": value_loss,
                }

            # Pre-update Gradient Evaluation against Exact E
            g_exact_E = extract_critic_flat_grads(jax.grad(exact_e_loss)(train_state.params))
            (_, init_losses), init_grads = jax.value_and_grad(E_loss_fn, has_aux=True)(
                train_state.params, traj_batch, targets
            )
            g_samp = extract_critic_flat_grads(init_grads)
            grad_mse_exact = jnp.sum((g_samp - g_exact_E) ** 2)
            grad_cos_exact = cos_similarity(g_samp, g_exact_E)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(E_loss_fn, has_aux=True)
                    (total_loss, losses), grads = grad_fn(train_state.params, traj_batch, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, losses

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), losses

            initial_update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, losses = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state

            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            metric.update({k: v.mean() for k, v in losses.items()})
            metric.update({"mean_rew": traj_batch.reward.mean()})
            value_metrics = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=True, light=config["LIGHT_METRICS"]
            )
            metric.update(value_metrics)
            if config["LOG_FEATURE_METRICS"]:
                metric.update(feature_metrics(
                    evaluator, network, train_state.params, random_policy=True,
                ))
            if hasattr(evaluator, "obs_stack") and hasattr(evaluator, "start_idx"):
                v_pred_start = network.apply(train_state.params, evaluator.obs_stack[evaluator.start_idx]).squeeze()
                metric["v_pred_start"] = v_pred_start

            # Gradient metrics
            metric["grad_mse_exact_E"] = grad_mse_exact
            metric["grad_cos_exact_E"] = grad_cos_exact
            metric["g_samp_norm"] = jnp.linalg.norm(g_samp)
            metric["g_exact_norm"] = jnp.linalg.norm(g_exact_E)

            runner_state = (train_state, env_state, last_obs, last_true_terminal, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, last_true_terminal, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
