# E_sample_v_aux_surgical.py
# Sampled Dirichlet Energy Minimization with Surgical Auxiliary Substitution (v_aux).
#
# Mathematical Foundation:
#   Under the Dirichlet energy decomposition:
#     E(theta) = (1 - gamma) ||e||_D^2 + 0.5 * gamma * sum_{i,j} mu_i P_{ij} (e_i - e_j)^2 + boundary_terms
#
#   Differentiating the Laplacian term 0.5 * (e_i - e_j)^2 with respect to theta yields four cross-terms:
#     grad_theta [ 0.5 * (e_i - e_j)^2 ] = -e_i * grad v_i  (Term 1)
#                                         + e_j * grad v_i  (Term 2)
#                                         - e_j * grad v_j  (Term 3)
#                                         + e_i * grad v_j  (Term 4, BIASED)
#
#   When evaluating with single-trajectory Monte Carlo returns G_t = r_t + gamma * G_{t+1}:
#   - Terms 1, 2, and 3 are unbiased under single trajectory sampling.
#   - ONLY Term 4 (+e_i * grad v_j) has double-sampling bias, because conditioning on s_{t+1} = j
#     correlates the trajectory return inside e_i with the next state's gradient grad v_j.
#
#   Surgical Formulation:
#   We use real Monte Carlo errors e_i = G_t - v_i, e_j = G_{t+1} - v_j in:
#     - (1 - gamma) magnitude anchor
#     - Terms 1, 2, and 3 of the Laplacian smoothing term
#     - Start-state anchor and timeout corrections
#   ONLY in Term 4 do we substitute the state-only auxiliary prediction e_tilde_i = sg(v_aux(s_t)) - v(s_t).
#
# Exact Loss Construction (No Manual Gradient Surgery Needed):
#   We construct the two-branch scalar loss:
#     L_Laplacian = 0.5 * (e_i - sg(e_j))^2 + 0.5 * (sg(e_tilde_i) - e_j)^2
#
#   Notice that autodiff differentiates this cleanly:
#     grad_{v_i} [ 0.5 * (e_i - sg(e_j))^2 ] = -(e_i - e_j) * grad v_i = -e_i * grad v_i + e_j * grad v_i (Terms 1 & 2)
#     grad_{v_j} [ 0.5 * (sg(e_tilde_i) - e_j)^2 ] = (e_tilde_i - e_j) * grad v_j = -e_j * grad v_j + e_tilde_i * grad v_j (Terms 3 & 4)
#
#   The sum of these two terms matches the exact surgical gradient analytically!

from core.imports import *
import core.helpers as helpers
import core.networks as networks
import distrax
import core.bellman_error as bellman_error
from core.feature_metrics import feature_metrics
import core.utils as utils

SAVE_DIR = "random/E_sample_v_aux_surgical"

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

    # --- Precompute True Value Function for Random Policy (if evaluator exists) ---
    if evaluator is not None:
        n_states = evaluator.num_states
        pi_active = jnp.ones((n_states, n_actions)) / n_actions
        terminal_policy = jnp.ones((1, n_actions)) / n_actions
        pi_full = jnp.vstack([pi_active, terminal_policy])
        V_true = evaluator.compute_true_values_raw(pi_full)
    else:
        n_states = None
        V_true = None

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config['k']
        # Initialize dual-critic network (shared torso, main head v, auxiliary head v_aux)
        network, network_params = networks.initialize_dual_critic_network(
            rng, obs_shape, env, env_params, k, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        last_true_terminal = jnp.zeros(config["NUM_ENVS"], dtype=bool)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, rng, idx = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                # Compute both values; use v_aux as baseline/continuation for MC GAE
                val, val_aux = network.apply(train_state.params, last_obs, method=network.both)
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
                next_val, next_val_aux = network.apply(train_state.params, true_next_obs, method=network.both)

                if "is_timeout" in info:
                    is_timeout = info["is_timeout"]
                else:
                    is_timeout = jnp.zeros_like(done, dtype=bool)
                true_terminal = done & ~is_timeout
                is_start = last_true_terminal

                transition = Transition(
                    done, action, val_aux, next_val_aux, reward, log_prob, last_obs, true_next_obs, 0.0, is_start, info
                )
                return (train_state, env_state, obsv, true_terminal, rng), transition

            env_step_state = (train_state, env_state, last_obs, last_true_terminal, rng)
            (_, env_state, last_obs, last_true_terminal, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

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

            # JOINT LOSS FUNCTION: Surgical Dirichlet E-loss + Auxiliary MC Loss
            def loss_fn(params, traj_batch, targets):
                gamma = config["GAMMA"]
                # Single forward pass through shared torso for both states
                v_i, v_aux_i = network.apply(params, traj_batch.obs, method=network.both)
                v_j, v_aux_j = network.apply(params, traj_batch.next_obs, method=network.both)
                
                true_term = traj_batch.done & ~traj_batch.info["is_timeout"]
                is_timeout = traj_batch.info["is_timeout"]
                aux_vf_coef = config.get("AUX_VF_COEF", 1.0)

                total_loss, magnitude_loss, laplacian_loss = helpers.e_critic_loss(
                    v_i, targets, v_j, traj_batch.next_target, true_term, gamma,
                    variant="surgical_v_aux",
                    is_start=traj_batch.is_start,
                    is_timeout=is_timeout,
                    v_aux_i=v_aux_i,
                    v_aux_j=v_aux_j,
                    aux_coef=aux_vf_coef,
                )

                aux_loss = jnp.mean((v_aux_i - targets) ** 2)
                e_loss = total_loss - aux_vf_coef * jnp.mean((jax.lax.stop_gradient(targets) - v_aux_i) ** 2)
                gap_v_v_aux = jnp.mean((v_i - jax.lax.stop_gradient(v_aux_i)) ** 2)

                return total_loss, {
                    "total_loss": total_loss,
                    "value_loss": e_loss,
                    "aux_loss": aux_loss,
                    "magnitude_loss": magnitude_loss,
                    "laplacian_loss": laplacian_loss,
                    "gap_v_v_aux": gap_v_v_aux,
                }

            # UPDATE NETWORK (Both heads + shared torso updated in the same backward pass)
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
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

            # Evaluator-based metrics (if evaluator is available)
            if evaluator is not None and V_true is not None:
                # Main value function metrics (evaluates v)
                value_metrics = bellman_error.value_metrics(
                    evaluator, network, train_state.params, random_policy=True, light=config["LIGHT_METRICS"]
                )
                metric.update(value_metrics)

                # Auxiliary value function evaluation (evaluates v_aux)
                v_aux_eval = network.apply(train_state.params, evaluator.obs_stack, method=network.v_aux)
                aux_value_error = jnp.mean((V_true[:n_states] - v_aux_eval) ** 2)
                metric["aux_value_error"] = aux_value_error

                # Main vs Auxiliary value function gap at evaluation states
                v_eval = network.apply(train_state.params, evaluator.obs_stack, method=network.value)
                metric["eval_gap_v_v_aux"] = jnp.mean((v_eval - v_aux_eval) ** 2)

                if config["LOG_FEATURE_METRICS"]:
                    metric.update(feature_metrics(
                        evaluator, network, train_state.params, random_policy=True,
                    ))
                if hasattr(evaluator, "obs_stack") and hasattr(evaluator, "start_idx"):
                    v_pred_start = network.apply(train_state.params, evaluator.obs_stack[evaluator.start_idx], method=network.value).squeeze()
                    v_aux_pred_start = network.apply(train_state.params, evaluator.obs_stack[evaluator.start_idx], method=network.v_aux).squeeze()
                    metric["v_pred_start"] = v_pred_start
                    metric["v_aux_pred_start"] = v_aux_pred_start

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
