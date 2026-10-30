# fixed_policy/sampled_E.py
# Unified Sampled E-loss Policy Evaluation for Fixed Pretrained Policy.
# Supports multiple loss variants via config:
#   - 'mc': Standard Monte Carlo target MSE regression
#   - 'oct8' / 'start_anchored': Oct 8 biased start-state anchored Dirichlet formulation
#   - 'surgical_v_aux': Surgical auxiliary substitution (with SHARED_TORSO=True/False)
#   - 'dirichlet_v_aux': Full auxiliary substitution
#   - 'sep14', 'oct5': Earlier uncorrected / boundary-corrected formulations

from core.imports import *
import core.helpers as helpers
import core.networks as networks
import distrax
import core.bellman_error as bellman_error
from core.feature_metrics import feature_metrics
import core.utils as utils

SAVE_DIR = "fixed/sampled_E"

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
    base_config = base_config.copy()
    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    base_config["NUM_MINIBATCHES"] = max(1, batch_size // base_config.get("MINIBATCH_SIZE", 256))
    base_config["NUM_UPDATES"] = base_config["TOTAL_TIMESTEPS"] // batch_size
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    try:
        n_actions = env.action_space(env_params).n
    except AttributeError:
        n_actions = env.action_space(env_params).shape[0]

    policy_fn, policy_matrix = helpers.get_evaluation_policies(base_config, evaluator)

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config.get("k", 32)
        loss_variant = config.get("LOSS_VARIANT", config.get("E_VARIANT", "oct8"))
        shared_torso = config.get("SHARED_TORSO", True)
        uses_aux = loss_variant in ["surgical_v_aux", "surgical", "dirichlet_v_aux", "bilinear_v_aux"]

        # Network Initialization: DualCritic if uses_aux, standard network otherwise
        if uses_aux:
            network, network_params = networks.initialize_dual_critic_network(
                rng, obs_shape, env, env_params, k, layer_norm=config.get("LAYER_NORM", True), shared_torso=shared_torso
            )
        else:
            network, network_params = networks.initialize_network(
                rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config.get("LAYER_NORM", True)
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
                value = network.apply(train_state.params, last_obs, method=network.value)
                pi = policy_fn(last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info["real_next_obs"]
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)

                if "is_timeout" in info:
                    is_timeout = info["is_timeout"]
                else:
                    is_timeout = jnp.zeros_like(done, dtype=bool)
                true_terminal = done & ~is_timeout
                is_start = last_true_terminal

                clean_info = {k_i: v_i for k_i, v_i in info.items() if k_i not in ["real_next_obs", "real_next_state"]}
                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, 0.0, is_start, clean_info
                )
                return (train_state, env_state, obsv, true_terminal, rng), transition

            env_step_state = (train_state, env_state, last_obs, last_true_terminal, rng)
            (_, env_state, last_obs, last_true_terminal, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # --- MC RETURN / TARGET CALCULATION ---
            return_lambda = config.get("RETURN_LAMBDA", config.get("VALUE_LAMBDA", 1.0))
            advantages, targets = helpers.calculate_gae(traj_batch, config["GAMMA"], return_lambda)
            
            if uses_aux:
                final_next_aux_val = network.apply(train_state.params, traj_batch.next_obs[-1], method=network.v_aux)
                traj_batch_aux = traj_batch._replace(next_value=traj_batch.next_value.at[-1].set(final_next_aux_val))
                aux_lambda = config.get("AUX_LAMBDA", 1.0)
                _, aux_targets = helpers.calculate_gae(traj_batch_aux, config["GAMMA"], aux_lambda)
            else:
                aux_targets = targets

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

            # LOSS FUNCTION
            def E_loss_fn(params, traj_batch, targets, aux_targets):
                gamma = config["GAMMA"]
                if "is_timeout" in traj_batch.info:
                    is_timeout_mb = traj_batch.info["is_timeout"]
                else:
                    is_timeout_mb = jnp.zeros_like(traj_batch.done, dtype=bool)
                true_term = traj_batch.done & ~is_timeout_mb

                if uses_aux:
                    v_i, v_aux_i = network.apply(params, traj_batch.obs, method=network.both)
                    v_j, v_aux_j = network.apply(params, traj_batch.next_obs, method=network.both)
                else:
                    v_i = network.apply(params, traj_batch.obs, method=network.value)
                    v_j = network.apply(params, traj_batch.next_obs, method=network.value)
                    v_aux_i, v_aux_j = None, None

                value_loss, magnitude_loss, laplacian_loss = helpers.e_critic_loss(
                    v_i, targets, v_j, traj_batch.next_target, true_term, gamma,
                    variant=loss_variant, is_start=traj_batch.is_start, is_timeout=is_timeout_mb,
                    v_aux_i=v_aux_i, v_aux_j=v_aux_j, aux_coef=config.get("AUX_VF_COEF", 1.0),
                    aux_targets_i=aux_targets if uses_aux else None,
                )
                
                total_loss = config.get("VF_COEF", 1.0) * value_loss
                return total_loss, {
                    "total_loss": total_loss,
                    "magnitude_loss": magnitude_loss,
                    "laplacian_loss": laplacian_loss,
                    "value_loss": value_loss,
                }

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets, aux_targets = batch_info
                    grad_fn = jax.value_and_grad(E_loss_fn, has_aux=True)
                    
                    (total_loss, losses), grads = grad_fn(
                        train_state.params, traj_batch, targets, aux_targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, losses

                train_state, traj_batch, advantages, targets, aux_targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets, aux_targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, aux_targets, rng), losses

            initial_update_state = (train_state, traj_batch, advantages, targets, aux_targets, rng)
            update_state, losses = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, _, rng = update_state

            # --------- Metrics ---------
            metric = {
                k_i: v_i.mean() 
                for k_i, v_i in traj_batch.info.items() 
                if k_i not in ["real_next_obs", "real_next_state"]
            }
            # Shared Metrics
            metric.update({k_l: v_l.mean() for k_l, v_l in losses.items()})
            metric.update({"mean_rew": traj_batch.reward.mean()})
            if config.get("CALC_TRUE_VALUES", False) and evaluator is not None:
                value_metrics = bellman_error.value_metrics(
                    evaluator, network, train_state.params, random_policy=False, target_policy_fn=policy_fn, light=config.get("LIGHT_METRICS", True)
                )
                metric.update(value_metrics)
                if config.get("LOG_FEATURE_METRICS", False):
                    metric.update(feature_metrics(
                        evaluator, network, train_state.params, random_policy=False, target_policy_fn=policy_fn
                    ))
                if hasattr(evaluator, "obs_stack") and hasattr(evaluator, "start_idx"):
                    v_pred_start = network.apply(train_state.params, evaluator.obs_stack[evaluator.start_idx], method=network.value).squeeze()
                    metric["v_pred_start"] = v_pred_start
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
