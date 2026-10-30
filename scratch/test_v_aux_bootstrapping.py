"""
scratch/test_v_aux_bootstrapping.py

Ablation test to determine whether v_aux should be bootstrapped using:
  1. 'boot_v':    The primary critic v (current codebase baseline)
  2. 'boot_self': The auxiliary critic v_aux itself (independent self-bootstrapping)
  3. 'fwd_only':  Forward-only (no v_aux)

Tested on:
  - fourrooms-mines (Sparse, lethal mines)
  - mountaincar (Sparse)
at lambda = 0.50.
"""

import os
import sys
import time
import argparse
from typing import NamedTuple, Optional, Dict, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from core.config import config as base_cfg
import core.helpers as helpers
import core.networks as networks
import core.runtime_metrics as runtime_metrics
import core.utils as utils


class CustomJumpTransition(NamedTuple):
    obs: jnp.ndarray
    dest_obs: jnp.ndarray
    target: jnp.ndarray
    dest_target: jnp.ndarray
    aux_target: jnp.ndarray
    hit_true_terminal: jnp.ndarray
    hit_timeout: jnp.ndarray
    start_weight: jnp.ndarray
    action: Optional[jnp.ndarray] = None
    log_prob: Optional[jnp.ndarray] = None
    advantage: Optional[jnp.ndarray] = None


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


def make_custom_train(base_config, mode="boot_v"):
    """
    mode in ["boot_v", "boot_self", "fwd_only"]
    """
    base_config = base_config.copy()
    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    base_config["NUM_MINIBATCHES"] = max(1, batch_size // base_config["MINIBATCH_SIZE"])
    base_config["NUM_UPDATES"] = max(1, base_config["TOTAL_TIMESTEPS"] // batch_size)
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config["k"]
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=2, layer_norm=config["LAYER_NORM"]
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)

        rng, _rng = jax.random.split(rng)
        aux_network, aux_params = networks.initialize_network(
            _rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config["LAYER_NORM"]
        )
        aux_train_state = networks.initialize_flax_train_state(config, aux_network, aux_params)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        start_obs = obsv
        steps_since_start = jnp.full((config["NUM_ENVS"],), 10_000, dtype=jnp.int32)

        def _update_step(runner_state, unused):
            train_state, aux_train_state, env_state, last_obs, steps_since_start, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)

                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    jax.random.split(_rng, config["NUM_ENVS"]), env_state, action, env_params
                )
                true_next_obs = info.get("real_next_obs", obsv)
                next_val = network.apply(train_state.params, obsv, method=network.value)

                clean_info = {k: v for k, v in info.items() if k not in ["real_next_obs", "real_next_state"]}
                transition = Transition(
                    done, action, value, next_val, reward, log_prob=pi.log_prob(action),
                    obs=last_obs, next_obs=true_next_obs, info=clean_info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. POLICY ADVANTAGES & PRIMARY RETURNS (using v)
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], config["GAE_LAMBDA"])
            _, returns_v = helpers.calculate_gae(traj_batch, config["GAMMA"], config["RETURN_LAMBDA"])

            # 3. COMPUTE AUX TARGET RETURNS
            if mode == "boot_self":
                # Self-bootstrapping for v_aux: evaluate aux_network on obs and horizon next_obs
                aux_vals = aux_network.apply(aux_train_state.params, traj_batch.obs)
                final_next_aux_val = aux_network.apply(aux_train_state.params, traj_batch.next_obs[-1])
                next_aux_vals = jnp.concatenate([aux_vals[1:], final_next_aux_val[None]], axis=0)

                traj_batch_aux = traj_batch._replace(value=aux_vals, next_value=next_aux_vals)
                _, returns_aux = helpers.calculate_gae(traj_batch_aux, config["GAMMA"], config["RETURN_LAMBDA"])
            else:
                # Baseline: v_aux bootstraps off primary critic v
                returns_aux = returns_v

            # 4. PREPARE JUMP PAIRS ONCE PER ROLLOUT
            rng, _rng = jax.random.split(rng)
            std_jump_batch, final_steps_since_start = helpers.prepare_e_lambda_jump_pairs(
                traj_batch=traj_batch,
                returns=returns_v,
                advantages=advantages,
                gamma=config["GAMMA"],
                lmbda=config["VALUE_LAMBDA"],
                rng=_rng,
                steps_since_start=steps_since_start,
                runway_length=config.get("RUNWAY_LENGTH", 16),
            )

            # Package with custom aux_target
            T_start = std_jump_batch.target.shape[0]
            jump_batch = CustomJumpTransition(
                obs=std_jump_batch.obs,
                dest_obs=std_jump_batch.dest_obs,
                target=std_jump_batch.target,
                dest_target=std_jump_batch.dest_target,
                aux_target=returns_aux[:T_start],
                hit_true_terminal=std_jump_batch.hit_true_terminal,
                hit_timeout=std_jump_batch.hit_timeout,
                start_weight=std_jump_batch.start_weight,
                action=std_jump_batch.action,
                log_prob=std_jump_batch.log_prob,
                advantage=std_jump_batch.advantage,
            )

            # 5. UPDATE NETWORK OVER EPOCHS AND FLAT MINIBATCHES
            variant_str = "forward" if mode == "fwd_only" else "aux"

            def _update_epoch(update_state, unused):
                def _update_minbatch(state, mb):
                    train_state, aux_train_state = state

                    # Adapt mb for e_lambda_ppo_minibatch_loss
                    def _ppo_loss_fn(p):
                        pi = network.apply(p, mb.obs, method=network.policy)
                        ratio = jnp.exp(pi.log_prob(mb.action) - mb.log_prob)
                        adv = helpers.post_process_advantage(mb.advantage, config)
                        loss_actor1 = ratio * adv
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        v_start = network.apply(p, mb.obs, method=network.value)
                        v_jump = network.apply(p, mb.dest_obs, method=network.value)

                        if mode != "fwd_only":
                            v_aux_start = aux_network.apply(aux_train_state.params, mb.obs)
                            v_aux_jump = aux_network.apply(aux_train_state.params, mb.dest_obs)
                        else:
                            v_aux_start = v_start
                            v_aux_jump = v_jump

                        value_loss, critic_losses = helpers.e_lambda_pointwise_critic_loss(
                            v_start=v_start,
                            v_jump=v_jump,
                            v_aux_start=v_aux_start,
                            v_aux_jump=v_aux_jump,
                            mb=mb,
                            gamma=config["GAMMA"],
                            lmbda=config["VALUE_LAMBDA"],
                            variant=variant_str,
                        )

                        tot_loss = (
                            config["POLICY_COEFF"] * loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return tot_loss, {
                            "total_loss": tot_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                        }

                    (tot_loss, losses), grads = jax.value_and_grad(_ppo_loss_fn, has_aux=True)(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)

                    # Update auxiliary critic on empirical returns (using mb.aux_target!)
                    if mode != "fwd_only":
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

            # 6. RUNTIME METRICS
            metric = runtime_metrics.compute_runtime_metrics(
                train_state=train_state,
                network=network,
                traj_batch=traj_batch,
                loss_info=loss_info,
                start_obs=start_obs,
                config=config,
                evaluator=evaluator,
            )

            runner_state = (train_state, aux_train_state, env_state, last_obs, final_steps_since_start, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, aux_train_state, env_state, obsv, steps_since_start, _rng, 1)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def run_benchmark():
    envs = ["fourrooms-mines", "mountaincar"]
    modes = ["boot_v", "boot_self", "fwd_only"]
    mode_names = {
        "boot_v": "E(0.50) [v_aux: boot from v] (Baseline)",
        "boot_self": "E(0.50) [v_aux: boot from self] (New)",
        "fwd_only": "E(0.50) [Forward-Only]",
    }

    n_seeds = 3
    num_updates = 60
    lmbda = 0.50
    critic_lr = 0.001
    actor_lr = 0.0003

    print("=" * 80)
    print("Ablation: How Should v_aux Be Bootstrapped at the Horizon?")
    print(f"Environments: {envs}")
    print(f"Lambda:       {lmbda}")
    print(f"Seeds:        {n_seeds}")
    print(f"Updates:      {num_updates}")
    print("=" * 80)

    rows = []

    for env_name in envs:
        print(f"\n>>> Running Environment: {env_name}")
        json_path = os.path.join(REPO_ROOT, "reference", "env_performance_baselines.json")
        import json
        with open(json_path, "r") as f:
            bdata = json.load(f).get(env_name, {"v_star": 1.0, "v_rand": 0.0, "delta": 1.0})
        v_star, v_rand, delta = float(bdata["v_star"]), float(bdata["v_rand"]), float(bdata["delta"])

        for mode in modes:
            print(f"  Testing {mode_names[mode]} ...", end="", flush=True)
            cfg = base_cfg.copy()
            cfg["ENV_NAME"] = env_name
            cfg["NUM_ENVS"] = 32
            cfg["NUM_STEPS"] = 128
            cfg["NUM_UPDATES"] = num_updates
            cfg["TOTAL_TIMESTEPS"] = num_updates * cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
            cfg["VALUE_LAMBDA"] = lmbda
            cfg["RETURN_LAMBDA"] = 1.0
            cfg["GAE_LAMBDA"] = 0.8
            cfg["ACTOR_LR"] = actor_lr
            cfg["CRITIC_LR"] = critic_lr
            cfg["CALC_TRUE_VALUES"] = False
            cfg["k"] = 32

            t0 = time.time()
            train_fn = make_custom_train(cfg, mode=mode)
            seeds = [42 + i * 100 for i in range(n_seeds)]
            rngs = jax.random.split(jax.random.PRNGKey(42), n_seeds)

            vmapped_train = jax.vmap(train_fn)
            out = vmapped_train(rngs)
            dt = time.time() - t0

            # Compute standardized metrics
            # metric is shaped (n_seeds, num_updates)
            metrics = out["metrics"]
            v_start = np.array(metrics["V_start"])  # (n_seeds, num_updates)
            norm_perf = (v_start - v_rand) / delta
            auc_perf = float(np.mean(np.trapz(norm_perf, axis=1) / (num_updates - 1)))
            final_perf_mean = float(np.mean(norm_perf[:, -1]))
            final_perf_sem = float(np.std(norm_perf[:, -1], ddof=1) / np.sqrt(n_seeds))

            ep_rets = np.array(metrics.get("returned_episode_returns", v_start))
            final_ep_ret = float(np.mean(ep_rets[:, -1]))
            ve_final = float(np.mean(metrics["nn_weighted_VE"][:, -1])) if "nn_weighted_VE" in metrics else float("nan")

            print(f" done ({dt:.1f}s) | AUC={auc_perf:+.4f} | Final_Vstart={final_perf_mean:+.4f} +/- {final_perf_sem:.4f} | Final_EpRet={final_ep_ret:.2f}")

            rows.append({
                "Environment": env_name,
                "Condition": mode_names[mode],
                "Mode": mode,
                "AUC Standardized Return": auc_perf,
                "Final Return Mean": final_perf_mean,
                "Final Return SEM": final_perf_sem,
                "Final Episode Return": final_ep_ret,
                "Final Value Error": ve_final,
                "Runtime (s)": dt,
            })

    res_df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("FINAL SUMMARY TABLE: BOOTSTRAPPING ABLATION (Lambda = 0.50)")
    print("=" * 90)
    print(res_df[["Environment", "Condition", "AUC Standardized Return", "Final Return Mean", "Final Return SEM", "Final Value Error"]].to_string(index=False))

    out_csv = os.path.join(REPO_ROOT, "results", "v_aux_bootstrap_ablation.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")


if __name__ == "__main__":
    run_benchmark()
