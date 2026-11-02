# Sampled version of PPO: TD(lambda) for critic, standard GAE for actor
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
import core.bellman_error as bellman_error
import core.runtime_metrics as runtime_metrics

SAVE_DIR = "ppo/td_lambda"

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
    base_config = base_config.copy()
    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    if "NUM_MINIBATCHES" in base_config:
        num_minibatches = base_config["NUM_MINIBATCHES"]
    else:
        minibatch_size = base_config.get("MINIBATCH_SIZE", 1024)
        num_minibatches = max(1, batch_size // minibatch_size)
    base_config["NUM_MINIBATCHES"] = num_minibatches
    base_config["NUM_UPDATES"] = base_config["TOTAL_TIMESTEPS"] // batch_size
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config.get("k", 32)

        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=2, layer_norm=config.get("LAYER_NORM", False)
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        start_obs = obsv

        runner_state = (train_state, env_state, obsv, rng, 1)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES
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
                true_next_obs = info["real_next_obs"]
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)

                clean_info = {k: v for k, v in info.items() if k not in ["real_next_obs", "real_next_state"]}
                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, clean_info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. ADVANTAGE AND TARGET CALCULATIONS
            # GAE_LAMBDA is strictly for policy advantages
            gae_lambda = config.get("GAE_LAMBDA", 0.95)
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], gae_lambda)

            # VALUE_LAMBDA is strictly for critic targets
            value_lambda = config.get("VALUE_LAMBDA", 0.95)
            _, targets = helpers.calculate_gae(traj_batch, config["GAMMA"], value_lambda)

            # 3. ACTOR UPDATE EPOCHS
            num_epochs_actor = config.get("NUM_EPOCHS_ACTOR", config.get("NUM_EPOCHS", 4))

            def _update_actor_epoch(update_state, unused):
                def _update_actor_minibatch(train_state, batch_info):
                    obs_mb, action_mb, log_prob_mb, advantages_mb = batch_info

                    def actor_loss_fn(params):
                        return helpers.ppo_actor_loss(
                            params, network, obs_mb, action_mb, log_prob_mb, advantages_mb, config
                        )

                    grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
                    (total_actor_loss, actor_metrics), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_actor_gradients(grads=grads)
                    return train_state, actor_metrics

                train_state, rng = update_state
                rng, _rng = jax.random.split(rng)
                actor_batch = (
                    traj_batch.obs,
                    traj_batch.action,
                    traj_batch.log_prob,
                    advantages,
                )
                minibatches = helpers.shuffle_and_batch(_rng, actor_batch, config["NUM_MINIBATCHES"])
                train_state, epoch_losses = jax.lax.scan(_update_actor_minibatch, train_state, minibatches)
                return (train_state, rng), epoch_losses

            if num_epochs_actor > 0:
                (train_state, rng), actor_loss_info = jax.lax.scan(
                    _update_actor_epoch, (train_state, rng), None, num_epochs_actor
                )
            else:
                actor_loss_info = {"actor_loss": jnp.array(0.0), "entropy": jnp.array(0.0)}

            # 4. CRITIC UPDATE EPOCHS (TD(lambda))
            num_epochs_critic = config.get("NUM_EPOCHS_CRITIC", config.get("NUM_EPOCHS", 4))

            def _update_critic_epoch(update_state, unused):
                def _update_critic_minibatch(train_state, batch_info):
                    obs_mb, targets_mb, value_mb = batch_info

                    def critic_loss_fn(params):
                        val_loss, val_metrics = helpers.critic_td_lambda_loss(
                            params, network, obs_mb, targets_mb, value_mb, config
                        )
                        scaled_loss = config.get("VF_COEF", 0.5) * val_loss
                        return scaled_loss, val_metrics

                    grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                    (scaled_v_loss, critic_metrics), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_critic_gradients(grads=grads)
                    return train_state, critic_metrics

                train_state, rng = update_state
                rng, _rng = jax.random.split(rng)
                critic_batch = (
                    traj_batch.obs,
                    targets,
                    traj_batch.value,
                )
                minibatches = helpers.shuffle_and_batch(_rng, critic_batch, config["NUM_MINIBATCHES"])
                train_state, epoch_losses = jax.lax.scan(_update_critic_minibatch, train_state, minibatches)
                return (train_state, rng), epoch_losses

            if num_epochs_critic > 0:
                (train_state, rng), critic_loss_info = jax.lax.scan(
                    _update_critic_epoch, (train_state, rng), None, num_epochs_critic
                )
            else:
                critic_loss_info = {"value_loss": jnp.array(0.0)}

            # 5. RUNTIME METRICS
            combined_loss_info = {
                "actor_loss": actor_loss_info["actor_loss"].mean(),
                "entropy": actor_loss_info["entropy"].mean(),
                "value_loss": critic_loss_info["value_loss"].mean(),
            }
            combined_loss_info["total_loss"] = (
                combined_loss_info["actor_loss"]
                + config.get("VF_COEF", 0.5) * combined_loss_info["value_loss"]
                - config.get("ENT_COEF", 0.01) * combined_loss_info["entropy"]
            )
            metric = runtime_metrics.compute_runtime_metrics(
                train_state=train_state,
                network=network,
                traj_batch=traj_batch,
                loss_info=combined_loss_info,
                start_obs=start_obs,
                config=config,
                evaluator=evaluator,
            )

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
