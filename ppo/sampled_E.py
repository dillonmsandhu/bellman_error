# ppo/sampled_E.py
# Unified Sampled E-loss PPO.
# Matches ppo/ppo.py structure: single shared loss updating actor and critic simultaneously.
# Supports multiple loss variants via config:
#   - 'mc': Standard Monte Carlo target MSE regression
#   - 'oct8' / 'start_anchored': Oct 8 biased start-state anchored Dirichlet formulation
#   - 'surgical_v_aux': Surgical auxiliary substitution (with SHARED_TORSO=True/False)
#   - 'dirichlet_v_aux': Full auxiliary substitution
#   - 'sep14', 'oct5': Earlier uncorrected / boundary-corrected formulations

from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
import core.bellman_error as bellman_error

SAVE_DIR = "ppo/sampled_E"

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    is_start: jnp.ndarray
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
        loss_variant = config.get("LOSS_VARIANT", config.get("E_VARIANT", "oct8"))
        shared_torso = config.get("SHARED_TORSO", True)
        uses_aux = loss_variant in ["surgical_v_aux", "surgical", "dirichlet_v_aux", "bilinear_v_aux"]

        # Network Initialization: DualCritic if uses_aux, standard network otherwise
        if uses_aux:
            network, network_params = networks.initialize_ac_dual_critic_network(
                rng, obs_shape, env, env_params, k, layer_norm=config.get("LAYER_NORM", False), shared_torso=shared_torso
            )
        else:
            network, network_params = networks.initialize_network(
                rng, obs_shape, env, env_params, k, n_heads=2, layer_norm=config.get("LAYER_NORM", False)
            )

        train_state = networks.initialize_flax_train_state(config, network, network_params)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        last_true_terminal = jnp.zeros(config["NUM_ENVS"], dtype=bool)

        runner_state = (train_state, env_state, obsv, last_true_terminal, rng, 1)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi = network.apply(train_state.params, last_obs, method=network.policy)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value = network.apply(train_state.params, last_obs, method=network.value)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info["real_next_obs"]
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)

                clean_info = {k_i: v_i for k_i, v_i in info.items() if k_i not in ["real_next_obs", "real_next_state"]}
                if "is_timeout" in info:
                    is_timeout = info["is_timeout"]
                else:
                    is_timeout = jnp.zeros_like(done, dtype=bool)
                true_terminal = done & ~is_timeout
                is_start = last_true_terminal
                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, is_start, clean_info
                )
                return (train_state, env_state, obsv, true_terminal, rng), transition

            env_step_state = (train_state, env_state, last_obs, last_true_terminal, rng)
            (_, env_state, last_obs, last_true_terminal, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. ADVANTAGE AND TARGET CALCULATIONS
            gae_lambda = config.get("GAE_LAMBDA", 0.8)
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], gae_lambda)

            return_lambda = config.get("RETURN_LAMBDA", config.get("VALUE_LAMBDA", 1.0))
            _, targets = helpers.calculate_gae(traj_batch, config["GAMMA"], return_lambda)

            if uses_aux:
                aux_lambda = config.get("AUX_LAMBDA", 0.9)
                _, aux_targets = helpers.calculate_gae(traj_batch, config["GAMMA"], aux_lambda)
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

            # 3. JOINT ACTOR-CRITIC LOSS (Matches ppo/ppo.py: single shared loss updating together)
            def _loss_fn(params, traj_batch_mb, advantages_mb, targets_mb, next_targets_mb, aux_targets_mb):
                # Actor Loss (PPO clipped surrogate)
                pi = network.apply(params, traj_batch_mb.obs, method=network.policy)
                log_prob = pi.log_prob(traj_batch_mb.action)
                entropy = pi.entropy().mean()
                ratio = jnp.exp(log_prob - traj_batch_mb.log_prob)

                adv_norm = helpers.post_process_advantage(advantages_mb, config)
                surr1 = ratio * adv_norm
                surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
                loss_actor = -jnp.minimum(surr1, surr2).mean()

                # Critic Loss
                if uses_aux:
                    v_i, v_aux_i = network.apply(params, traj_batch_mb.obs, method=network.both)
                    v_j, v_aux_j = network.apply(params, traj_batch_mb.next_obs, method=network.both)
                else:
                    v_i = network.apply(params, traj_batch_mb.obs, method=network.value)
                    v_j = network.apply(params, traj_batch_mb.next_obs, method=network.value)
                    v_aux_i, v_aux_j = None, None

                if "is_timeout" in traj_batch_mb.info:
                    is_timeout_mb = traj_batch_mb.info["is_timeout"]
                else:
                    is_timeout_mb = jnp.zeros_like(traj_batch_mb.done, dtype=bool)
                true_term_mb = traj_batch_mb.done & ~is_timeout_mb

                value_loss, magnitude_loss, laplacian_loss = helpers.e_critic_loss(
                    v_i, targets_mb, v_j, next_targets_mb, true_term_mb, config["GAMMA"],
                    variant=loss_variant, is_start=traj_batch_mb.is_start, is_timeout=is_timeout_mb,
                    v_aux_i=v_aux_i, v_aux_j=v_aux_j, aux_coef=config.get("AUX_VF_COEF", 1.0),
                    aux_targets_i=aux_targets_mb if uses_aux else None,
                )

                total_loss = (
                    config.get("POLICY_COEFF", 1.0) * loss_actor
                    + config.get("VF_COEF", 0.5) * value_loss
                    - config.get("ENT_COEF", 0.01) * entropy
                )
                return total_loss, (total_loss, value_loss, loss_actor, entropy, magnitude_loss, laplacian_loss)

            # 4. UPDATE EPOCHS & MINIBATCHES
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch_mb, advantages_mb, targets_mb, next_targets_mb, aux_targets_mb = batch_info
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, loss_tuple), grads = grad_fn(
                        train_state.params, traj_batch_mb, advantages_mb, targets_mb, next_targets_mb, aux_targets_mb
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_tuple

                train_state, traj_batch, advantages, targets, next_targets, aux_targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets, next_targets, aux_targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                train_state, loss_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, next_targets, aux_targets, rng), loss_info

            initial_update_state = (train_state, traj_batch, advantages, targets, next_targets, aux_targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, _, _, rng = update_state

            # 5. METRICS
            metric = {
                k_i: v_i.mean() 
                for k_i, v_i in traj_batch.info.items() 
                if k_i not in ["real_next_obs", "real_next_state"]
            }
            metric.update({
                "total_loss": loss_info[0].mean(),
                "value_loss": loss_info[1].mean(),
                "actor_loss": loss_info[2].mean(),
                "entropy": loss_info[3].mean(),
                "magnitude_loss": loss_info[4].mean(),
                "laplacian_loss": loss_info[5].mean(),
                "mean_rew": traj_batch.reward.mean(),
            })

            value_metrics = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=False, light=config.get("LIGHT_METRICS", True)
            )
            metric.update(value_metrics)
            if hasattr(evaluator, "obs_stack") and hasattr(evaluator, "start_idx"):
                v_pred_start = network.apply(train_state.params, evaluator.obs_stack[evaluator.start_idx], method=network.value).squeeze()
                metric["v_pred_start"] = v_pred_start

            runner_state = (train_state, env_state, obsv, last_true_terminal, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, last_true_terminal, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
