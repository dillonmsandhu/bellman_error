# random_policy/rotation_gated_E.py
# Rotation-Gated Sampled Dirichlet Energy Minimization (E_sample_v_aux_surgical + online S/K gate).
#
# Mathematical Foundation:
#   In the surgical v_aux loss (see reference/paper/main.tex, Eq. full_sample_grad):
#     l_i(theta; omega) = (1 - gamma) * e_i^2
#                       + 0.5 * gamma * (e_i - sg(e_j))^2             [Term 2: forward branch]
#                       + omega * 0.5 * gamma * (sg(e_tilde_i) - e_tilde_j)^2  [Term 1: backward v_aux branch]
#
#   Taking the expected negative gradient under stationarity gives:
#     -E[grad_theta l_i(theta; omega)] = (1 + omega) J^T S e
#                                      + (1 - omega) J^T K e
#                                      + (1 - omega)(1 - gamma) J^T D e.
#
#   - When omega = 0: exact "forward_only" update (zero v_aux bias/variance on the critic).
#   - When omega = 1: exact "surgical_v_aux" E(0) update (cancels K completely).
#
#   Online Rotation Detector (1 JVP + 1 VJP per rollout, zero v_aux):
#     Along the unit TD(0) semi-gradient direction u = g_TD / ||g_TD||_2:
#       alpha_eff = u^T S_J u  = mean(j_s^2 - gamma * m * j_s * j_sp)
#       K_J u     = (gamma / 2B) * (VJP_sp(m * j_s) - VJP_s(m * j_sp))
#       beta_eff  = ||K_J u||_2
#     With separate EMAs alpha_ema, beta_ema and r_rot_ema = beta_ema / max(alpha_ema, 1e-6):
#       omega_t   = max(0, (r_rot_ema - tau) / (r_rot_ema + tau)) in [0, 1),
#     derived from the worst-case contraction margin ((1 - omega) / (1 + omega)) * r_rot <= tau.

from core.imports import *
import core.helpers as helpers
import core.networks as networks
import distrax
import core.bellman_error as bellman_error
from core.feature_metrics import feature_metrics
import core.utils as utils
from jax.flatten_util import ravel_pytree

SAVE_DIR = "random/rotation_gated_E"


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


def compute_online_rotation_ratio(network, params, obs, next_obs, rewards, true_terminal, gamma):
    """
    Computes online Rayleigh-quotient rotation ratio r_rot = beta_eff / alpha_eff
    along the sampled TD(0) semi-gradient direction u = g_TD / ||g_TD||_2 using 1 JVP + 1 VJP.
    Zero v_aux dependency, zero double-sampling bias.
    """
    not_term = (~true_terminal).astype(jnp.float32)
    B = obs.shape[0]

    def v_s_fn(p):
        return network.apply(p, obs, method=network.value)

    def v_sp_fn(p):
        return network.apply(p, next_obs, method=network.value)

    def td_loss_fn(p):
        v_s = v_s_fn(p)
        v_sp = v_sp_fn(p)
        target = jax.lax.stop_gradient(rewards + gamma * not_term * v_sp)
        return 0.5 * jnp.mean((v_s - target) ** 2)

    g_td = jax.grad(td_loss_fn)(params)
    g_flat, unravel = ravel_pytree(g_td)
    g_norm = jnp.linalg.norm(g_flat)
    u_tree = unravel(g_flat / jnp.maximum(g_norm, 1e-12))

    # Forward-mode JVPs: j_s = J_s u, j_sp = J_sp u
    _, j_s = jax.jvp(v_s_fn, (params,), (u_tree,))
    _, j_sp = jax.jvp(v_sp_fn, (params,), (u_tree,))

    # alpha_eff = u^T S_J u = (1/B) sum_b [j_s(b)^2 - gamma * not_term_b * j_s(b) * j_sp(b)]
    alpha_eff = jnp.mean(j_s * j_s - gamma * not_term * j_s * j_sp)

    # K_J u = (gamma / 2B) * (J_sp^T (not_term * j_s) - J_s^T (not_term * j_sp))
    _, vjp_s = jax.vjp(v_s_fn, params)
    _, vjp_sp = jax.vjp(v_sp_fn, params)
    w_for_sp = (0.5 * gamma / B) * (not_term * j_s)
    w_for_s = (0.5 * gamma / B) * (not_term * j_sp)
    ku_1 = vjp_sp(w_for_sp)[0]
    ku_2 = vjp_s(w_for_s)[0]
    ku_tree = jax.tree_util.tree_map(lambda a, b: a - b, ku_1, ku_2)
    ku_flat, _ = ravel_pytree(ku_tree)
    beta_eff = jnp.linalg.norm(ku_flat)

    r_rot = beta_eff / jnp.maximum(alpha_eff, 1e-8)
    return alpha_eff, beta_eff, r_rot


def make_train(base_config):
    base_config = base_config.copy()
    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    minibatch_size = base_config["MINIBATCH_SIZE"]
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
        k = config["k"]
        rot_gate_mode = config["ROT_GATE_MODE"]
        if rot_gate_mode not in ("adaptive", "forward_only", "full_e"):
            raise ValueError(f"Unrecognized ROT_GATE_MODE '{rot_gate_mode}'")
        rot_gate_tau = float(config["ROT_GATE_TAU"])
        if not (0.0 < rot_gate_tau <= 1.0):
            raise ValueError(f"ROT_GATE_TAU must be in (0, 1], got {rot_gate_tau}")
        rot_gate_ema = float(config["ROT_GATE_EMA"])

        network, network_params = networks.initialize_dual_critic_network(
            rng,
            obs_shape,
            env,
            env_params,
            k,
            layer_norm=config["LAYER_NORM"],
            shared_torso=config["SHARED_TORSO"],
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        last_true_terminal = jnp.zeros(config["NUM_ENVS"], dtype=bool)
        alpha_ema_init = jnp.asarray(0.0, dtype=jnp.float32)
        beta_ema_init = jnp.asarray(0.0, dtype=jnp.float32)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, alpha_ema, beta_ema, rng, idx = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                val, val_aux = network.apply(train_state.params, last_obs, method=network.both)
                if is_continuous:
                    pi = distrax.Independent(
                        distrax.Uniform(
                            low=jnp.full((config["NUM_ENVS"], *action_shape), -1.0),
                            high=jnp.full((config["NUM_ENVS"], *action_shape), 1.0),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                else:
                    pi = distrax.Categorical(logits=jnp.zeros((config["NUM_ENVS"], n_actions)))

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info["real_next_obs"]
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

            # MC Return calculation (GAE lambda=RETURN_LAMBDA)
            return_lambda = config["RETURN_LAMBDA"]
            advantages, targets = helpers.calculate_gae(traj_batch, config["GAMMA"], return_lambda)

            if "is_timeout" in traj_batch.info:
                is_timeout = traj_batch.info["is_timeout"]
            else:
                is_timeout = jnp.zeros_like(traj_batch.done, dtype=bool)
            true_terminal = traj_batch.done & ~is_timeout

            # Align next targets G_{t+1}
            next_targets = jnp.roll(targets, shift=-1, axis=0)
            next_targets = next_targets.at[-1].set(traj_batch.next_value[-1])
            next_targets = jnp.where(is_timeout, traj_batch.next_value, next_targets)
            next_targets = jnp.where(true_terminal, 0.0, next_targets)
            traj_batch = traj_batch._replace(next_target=next_targets)

            # --- ONLINE ROTATION DETECTOR (once per rollout, up to ROT_PROBE_BATCH transitions) ---
            flat_obs = traj_batch.obs.reshape((-1,) + obs_shape)
            flat_next_obs = traj_batch.next_obs.reshape((-1,) + obs_shape)
            flat_rew = traj_batch.reward.reshape((-1,))
            flat_term = true_terminal.reshape((-1,))
            n_probe = min(int(config["ROT_PROBE_BATCH"]), flat_obs.shape[0])
            alpha_eff, beta_eff, r_rot = compute_online_rotation_ratio(
                network,
                train_state.params,
                flat_obs[:n_probe],
                flat_next_obs[:n_probe],
                flat_rew[:n_probe],
                flat_term[:n_probe],
                config["GAMMA"],
            )
            alpha_ema = jnp.where(idx == 1, alpha_eff, rot_gate_ema * alpha_ema + (1.0 - rot_gate_ema) * alpha_eff)
            beta_ema = jnp.where(idx == 1, beta_eff, rot_gate_ema * beta_ema + (1.0 - rot_gate_ema) * beta_eff)
            r_rot_ema = beta_ema / jnp.maximum(alpha_ema, 1e-6)

            if rot_gate_mode == "adaptive":
                omega = jnp.maximum(0.0, (r_rot_ema - rot_gate_tau) / (r_rot_ema + rot_gate_tau))
            elif rot_gate_mode == "forward_only":
                omega = jnp.asarray(0.0, dtype=jnp.float32)
            elif rot_gate_mode == "full_e":
                omega = jnp.asarray(1.0, dtype=jnp.float32)

            # JOINT LOSS FUNCTION: Rotation-Gated Surgical Dirichlet E-loss + Auxiliary MC Loss
            def loss_fn(params, traj_batch, targets):
                gamma = config["GAMMA"]
                sg = jax.lax.stop_gradient
                v_i, v_aux_i = network.apply(params, traj_batch.obs, method=network.both)
                v_j, v_aux_j = network.apply(params, traj_batch.next_obs, method=network.both)

                if "is_timeout" in traj_batch.info:
                    is_timeout_mb = traj_batch.info["is_timeout"]
                else:
                    is_timeout_mb = jnp.zeros_like(traj_batch.done, dtype=bool)
                true_term = traj_batch.done & ~is_timeout_mb
                aux_vf_coef = config["AUX_VF_COEF"]

                e_i = targets - v_i
                e_j = jnp.where(true_term, 0.0, traj_batch.next_target - v_j)
                e_tilde_i = sg(v_aux_i) - v_i
                e_tilde_j = jnp.where(true_term, 0.0, sg(v_aux_j) - v_j)

                weight_mc = 1.0 - gamma
                weight_dir = 0.5 * gamma

                magnitude_loss = weight_mc * jnp.mean(e_i ** 2)
                branch_v_i = (e_i - sg(e_j)) ** 2
                branch_v_j = jnp.where(true_term, 0.0, (sg(e_tilde_i) - e_tilde_j) ** 2)
                laplacian_loss = weight_dir * jnp.mean(branch_v_i + omega * branch_v_j)

                start_anchor_loss = weight_dir * jnp.mean(jnp.where(traj_batch.is_start, e_i ** 2, 0.0))
                timeout_corr_loss = -weight_dir * jnp.mean(jnp.where(is_timeout_mb, e_j ** 2, 0.0))

                aux_loss = jnp.mean((sg(targets) - v_aux_i) ** 2)
                e_loss = magnitude_loss + laplacian_loss + start_anchor_loss + timeout_corr_loss
                total_loss = e_loss + aux_vf_coef * aux_loss
                gap_v_v_aux = jnp.mean((v_i - sg(v_aux_i)) ** 2)

                return total_loss, {
                    "total_loss": total_loss,
                    "value_loss": e_loss,
                    "aux_loss": aux_loss,
                    "magnitude_loss": magnitude_loss,
                    "laplacian_loss": laplacian_loss,
                    "gap_v_v_aux": gap_v_v_aux,
                }

            # UPDATE NETWORK
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
                k_i: v_i.mean()
                for k_i, v_i in traj_batch.info.items()
                if k_i not in ["real_next_obs", "real_next_state"]
            }
            metric.update({k_l: v_l.mean() for k_l, v_l in losses.items()})
            metric.update({
                "mean_rew": traj_batch.reward.mean(),
                "rot_omega": omega,
                "rot_r_rot": r_rot,
                "rot_r_rot_ema": r_rot_ema,
                "rot_alpha_eff": alpha_eff,
                "rot_beta_eff": beta_eff,
            })

            if evaluator is not None and V_true is not None:
                value_metrics = bellman_error.value_metrics(
                    evaluator, network, train_state.params, random_policy=True, light=config["LIGHT_METRICS"]
                )
                metric.update(value_metrics)

                v_aux_eval = network.apply(train_state.params, evaluator.obs_stack, method=network.v_aux)
                metric["aux_value_error"] = jnp.mean((V_true[:n_states] - v_aux_eval) ** 2)

                v_eval = network.apply(train_state.params, evaluator.obs_stack, method=network.value)
                metric["eval_gap_v_v_aux"] = jnp.mean((v_eval - v_aux_eval) ** 2)

                if config["LOG_FEATURE_METRICS"]:
                    metric.update(feature_metrics(
                        evaluator, network, train_state.params, random_policy=True,
                    ))
                if hasattr(evaluator, "obs_stack") and hasattr(evaluator, "start_idx"):
                    metric["v_pred_start"] = network.apply(
                        train_state.params, evaluator.obs_stack[evaluator.start_idx], method=network.value
                    ).squeeze()
                    metric["v_aux_pred_start"] = network.apply(
                        train_state.params, evaluator.obs_stack[evaluator.start_idx], method=network.v_aux
                    ).squeeze()

            runner_state = (train_state, env_state, last_obs, last_true_terminal, alpha_ema, beta_ema, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, last_true_terminal, alpha_ema_init, beta_ema_init, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
