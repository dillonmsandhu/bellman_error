# Sampled E(lambda) with Fixed/Stop-Gradient Error Traces (Fixed Policy)
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import distrax
import core.bellman_error as bellman_error
from core.feature_metrics import feature_metrics
import core.utils as utils

SAVE_DIR = "fixed/E_lambda_fixed"

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def make_train(base_config):
    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    base_config["NUM_MINIBATCHES"] = max(1, batch_size // base_config.get("MINIBATCH_SIZE", batch_size))
    base_config["NUM_UPDATES"] = max(1, base_config["TOTAL_TIMESTEPS"] // batch_size)
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n

    policy_fn, policy_matrix = helpers.get_evaluation_policies(base_config, evaluator)

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config.get('k', 32)
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                value = network.apply(train_state.params, last_obs)
                pi = policy_fn(last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info['real_next_obs']
                next_val = network.apply(train_state.params, true_next_obs)

                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # 2. TARGET CALCULATION
            e_lambda = config.get("VALUE_LAMBDA", config.get("E_LAMBDA", 0.0))
            return_lambda = config.get("RETURN_LAMBDA", 1.0)
            targets, _ = helpers.calculate_e_lambda_targets(
                traj_batch, config["GAMMA"], e_lambda, return_lambda
            )

            # 3. UPDATE NETWORK OVER EPOCHS AND MINIBATCHES
            recompute_targets = config.get("RECOMPUTE_TARGETS_EACH_EPOCH", False)

            def _update_epoch(update_state, unused):
                train_state, traj_batch, targets, rng = update_state

                def _do_recompute(_):
                    fresh_values = network.apply(train_state.params, traj_batch.obs)
                    fresh_v_T = network.apply(train_state.params, traj_batch.next_obs[-1])
                    traj_fresh = traj_batch._replace(value=fresh_values)
                    fresh_targets, _ = helpers.calculate_e_lambda_targets(
                        traj_fresh, config["GAMMA"], e_lambda, return_lambda, next_value_T=fresh_v_T
                    )
                    return traj_fresh, fresh_targets

                def _no_recompute(_):
                    return traj_batch, targets

                traj_batch_fresh, targets = jax.lax.cond(
                    jnp.array(recompute_targets, dtype=bool),
                    _do_recompute,
                    _no_recompute,
                    operand=None,
                )

                def _update_minbatch(train_state, batch_info):
                    traj_batch_mb, targets_mb = batch_info
                    def _loss_fn(params):
                        v_pred = network.apply(params, traj_batch_mb.obs)
                        val_loss = 0.5 * jnp.mean(jnp.square(v_pred - targets_mb))
                        tot_loss = config.get("VF_COEF", 1.0) * val_loss
                        return tot_loss, {
                            "total_loss": tot_loss,
                            "value_loss": val_loss,
                        }

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, losses), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, losses

                rng, _rng = jax.random.split(rng)
                batch = (traj_batch_fresh, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch_fresh, targets, rng), losses

            initial_update_state = (train_state, traj_batch, targets, rng)
            update_state, losses = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, rng = update_state

            # 4. METRICS
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            metric.update({k: v.mean() for k, v in losses.items()})
            metric.update({
                "mean_rew": traj_batch.reward.mean(),
                "v_pred": traj_batch.value.mean(),
            })

            if evaluator is not None and config.get("CALC_TRUE_VALUES", False):
                value_metrics = bellman_error.value_metrics(
                    evaluator, network, train_state.params, random_policy=False, target_policy_fn=policy_fn, light=config.get("LIGHT_METRICS", True)
                )
                metric.update(value_metrics)
                if config.get("LOG_FEATURE_METRICS", False):
                    metric.update(feature_metrics(
                        evaluator, network, train_state.params, random_policy=False, target_policy_fn=policy_fn,
                    ))
                if hasattr(evaluator, "obs_stack") and hasattr(evaluator, "start_idx"):
                    v_pred_start = network.apply(train_state.params, evaluator.obs_stack[evaluator.start_idx]).squeeze()
                    metric["v_pred_start"] = v_pred_start

            runner_state = (train_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
