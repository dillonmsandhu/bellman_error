# Sampled E(lambda) with Geometric Jump Sampling (Random Policy)
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import distrax
import core.bellman_error as bellman_error
from core.feature_metrics import feature_metrics
import core.utils as utils

SAVE_DIR = "random/E_lambda_geometric"

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
    base_config["NUM_MINIBATCHES"] = max(1, batch_size // base_config["MINIBATCH_SIZE"])
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

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config['k']
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)

        rng, _rng = jax.random.split(rng)
        aux_network, aux_params = networks.initialize_network(
            _rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        aux_train_state = networks.initialize_flax_train_state(config, aux_network, aux_params)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        steps_since_start = jnp.full((config["NUM_ENVS"],), 10_000, dtype=jnp.int32)

        def _update_step(runner_state, unused):
            train_state, aux_train_state, env_state, last_obs, steps_since_start, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

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

                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # 2. BASELINE RETURNS (MC returns G_t)
            return_lambda = config["RETURN_LAMBDA"]
            _, returns = helpers.calculate_gae(traj_batch, config["GAMMA"], return_lambda)

            # 3. SELF-BOOTSTRAPPED AUXILIARY RETURNS
            variant = config.get("E_LAMBDA_VARIANT", config.get("E_VARIANT", "aux"))
            if variant in ["aux", "v_aux"]:
                aux_returns = helpers.calculate_self_bootstrapped_aux_returns(
                    aux_network=aux_network,
                    aux_params=aux_train_state.params,
                    traj_batch=traj_batch,
                    gamma=config["GAMMA"],
                    return_lambda=return_lambda,
                )
            else:
                aux_returns = returns

            # 4. PREPARE JUMP PAIRS ONCE PER ROLLOUT
            rng, _rng = jax.random.split(rng)
            jump_batch, final_steps_since_start = helpers.prepare_e_lambda_jump_pairs(
                traj_batch=traj_batch,
                returns=returns,
                gamma=config["GAMMA"],
                lmbda=config["VALUE_LAMBDA"],
                rng=_rng,
                steps_since_start=steps_since_start,
                runway_length=config.get("RUNWAY_LENGTH", 16),
                aux_returns=aux_returns,
            )

            # 5. UPDATE NETWORK OVER EPOCHS AND FLAT MINIBATCHES
            def _update_epoch(update_state, unused):
                def _update_minbatch(state, mb):
                    train_state, aux_train_state = state

                    grad_fn = jax.value_and_grad(helpers.e_lambda_fixed_critic_minibatch_loss, has_aux=True)
                    (total_loss, losses), grads = grad_fn(
                        train_state.params,
                        network,
                        mb,
                        config,
                        aux_params=aux_train_state.params,
                        aux_network=aux_network,
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    # Update auxiliary critic on self-bootstrapped empirical returns
                    if variant in ["aux", "v_aux"]:
                        def _aux_loss_fn(aux_p):
                            aux_vals = aux_network.apply(aux_p, mb.obs)
                            return 0.5 * jnp.mean((jax.lax.stop_gradient(mb.aux_target) - aux_vals) ** 2)

                        aux_loss, aux_grads = jax.value_and_grad(_aux_loss_fn)(aux_train_state.params)
                        aux_train_state = aux_train_state.apply_gradients(grads=aux_grads)
                        losses["aux_loss"] = aux_loss
                    else:
                        losses["aux_loss"] = 0.0

                    return (train_state, aux_train_state), losses

                train_state, aux_train_state, rng = update_state
                rng, _rng = jax.random.split(rng)
                minibatches = helpers.shuffle_and_batch(_rng, jump_batch, config["NUM_MINIBATCHES"])
                (train_state, aux_train_state), loss_info = jax.lax.scan(
                    _update_minbatch, (train_state, aux_train_state), minibatches
                )
                return (train_state, aux_train_state, rng), loss_info

            initial_update_state = (train_state, aux_train_state, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, aux_train_state, rng = update_state

            # 4. METRICS
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            metric.update({k: v.mean() for k, v in loss_info.items()})
            metric.update({
                "mean_rew": traj_batch.reward.mean(),
                "v_pred": traj_batch.value.mean(),
            })

            if evaluator is not None and config["CALC_TRUE_VALUES"]:
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

            runner_state = (train_state, aux_train_state, env_state, last_obs, final_steps_since_start, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, aux_train_state, env_state, obsv, steps_since_start, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
