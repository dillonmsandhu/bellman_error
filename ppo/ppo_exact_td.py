from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
from flax.training.train_state import TrainState
import core.bellman_error as bellman_error

SAVE_DIR = "ppo/exact_td"

def network_inference(params, network, S, n_actions):
    pi_dist, v = network.apply(params, S)
    pi = pi_dist.probs
    terminal_policy = jnp.ones([1, n_actions], dtype=pi.dtype) / n_actions
    pi = jnp.vstack([pi, terminal_policy])
    v = jnp.append(v, 0.0)
    return pi, v

def make_train(config):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"]
    config['NUM_ENVS'] = 1
    config['NUM_STEPS'] = 1
    
    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n
    
    S = evaluator.obs_stack
    P = evaluator.P 

    def train(rng, hparams=None):
        if hparams is None:
            hparams = {}

        lr = hparams.get('LR', config['LR'])
        lr_end = hparams.get('LR_END', config.get('LR_END', lr))
        weight_decay = hparams.get('WEIGHT_DECAY', config.get('WEIGHT_DECAY', 1e-2))
        adam_eps = hparams.get('ADAM_EPS', config.get('ADAM_EPS', 1e-4))
        max_grad_norm = hparams.get('MAX_GRAD_NORM', config.get('MAX_GRAD_NORM', 1.0))
        γ = hparams.get('GAMMA', config['GAMMA'])

        k = config.get('k', 32)
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=2, layer_norm=config['LAYER_NORM']
        )
        total_grad_steps = config["NUM_UPDATES"] * config["NUM_EPOCHS"]
        lr_scheduler = optax.linear_schedule(lr, lr_end, total_grad_steps)
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adamw(lr_scheduler, weight_decay=weight_decay, eps=adam_eps),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        runner_state = (train_state, 1)

        def _update_step(runner_state, unused):
            train_state, idx = runner_state

            # 1. Compute Exact Dynamics
            old_pi_dist, old_v = network.apply(train_state.params, S)
            old_pi = old_pi_dist.probs
            terminal_policy = jnp.ones([1, n_actions], dtype=old_pi.dtype) / n_actions
            old_pi_full = jnp.vstack([old_pi, terminal_policy])
            old_log_pi = jnp.log(old_pi + 1e-8)
            
            old_v_full = jnp.append(old_v, 0.0)

            P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)
            R_pi = jnp.einsum("sa,sam,sam->s", old_pi_full, P, evaluator.R)
            
            mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
            mu = jnp.append(mu, 0.0)

            # 2. Compute Advantages
            R_sa = jnp.einsum("sam,sam->sa", P[:-1], evaluator.R[:-1])
            Q_sa = R_sa + γ * jnp.einsum("sam,m->sa", P[:-1], old_v_full)
            
            A = Q_sa - old_v[:, None]
            A = jax.lax.stop_gradient(A)

            def loss_fn(params, network):
                # A shape is (num_states, num_actions)
                pi, v = network_inference(params, network, S, n_actions)
                TD_targets = R_pi + γ * P_pi @ v
                
                # Value Loss
                td_errors = v - jax.lax.stop_gradient(TD_targets)
                value_loss = 0.5 * jnp.sum(mu * (td_errors ** 2)) * config.get("VF_COEF", 0.5)

                # Policy Loss
                log_pi = jnp.log(pi[:-1, :] + 1e-8)
                log_pi_sum = jnp.sum(pi[:-1, :] * log_pi, axis=-1)
                entropy = -jnp.sum(mu[:-1] * log_pi_sum) * config.get("ENT_COEF", 0.01)
                # entropy = -jnp.sum(mu[:-1] * log_pi_sum) * 0.0

                # PPO clip loss
                ratio = jnp.exp(log_pi - old_log_pi)
                pi_old = jnp.exp(old_log_pi)
    
                surr1 = ratio * A
                surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * A
                
                actor_loss = -jnp.sum(mu[:-1, None] * pi_old * jnp.minimum(surr1, surr2))

                total_loss = value_loss + actor_loss - entropy
                return total_loss, (value_loss, actor_loss, entropy, v)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            # 3. Apply expected update NUM_EPOCHS times
            def epoch_step(train_state, unused):
                (total_loss, metrics), grads = grad_fn(train_state.params, network)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, metrics

            train_state, epoch_metrics = jax.lax.scan(epoch_step, train_state, None, config["NUM_EPOCHS"])
            
            # Metrics
            value_loss, actor_loss, entropy, v_pred = epoch_metrics
            metric = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=False
            )
            if config.get("LOG_FEATURE_METRICS", False):
                from core.feature_metrics import feature_metrics
                metric.update(feature_metrics(
                    evaluator, network, train_state.params, random_policy=False,)
                )
            metric.update({
                "total_loss": (value_loss + actor_loss - entropy).mean(),
                "value_loss": value_loss.mean(),
                "actor_loss": actor_loss.mean(),
                "entropy": entropy.mean(),
                "v_pred_start": v_pred.squeeze()[evaluator.start_idx],
                "Mean_A": A.mean(),
            })
            
            runner_state = (train_state, idx + 1)
            return runner_state, metric

        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
