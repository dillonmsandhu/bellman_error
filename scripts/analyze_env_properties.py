import os
import sys
import json
import time
import argparse
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
from core.config import config as base_cfg
import core.helpers as helpers
import core.networks as networks
from ppo.ground_truth import make_train as make_gt_train

ALL_ENVS = [
    "FourRooms-misc",
    "fourrooms-dense",
    "fourrooms-mines",
    "fourrooms-mines-dense",
    "eightrooms-misc",
    "eightrooms-dense",
    "whirlpool-misc",
    "MountainCar-v0",
    "mountaincar-dense",
    "SpaceInvadersExactValue",
]

def analyze_env(env_name: str, gt_updates: int = 100, seed: int = 42):
    """
    1. Trains policy using ppo/ground_truth.py for `gt_updates` updates.
    2. Extracts policy pi(a|s) and stationary distribution mu(s).
    3. Computes:
       - Transition Asymmetry: ||K||_2 / ||S||_2
       - Reward Density: sum_{s: |R(s)|>0} mu(s)
       - State Visitation Entropy: -sum mu log mu
       - Terminal Flux: mu(terminals)
       - Sample Gradient Variance Ratio: Var(Grad E) / Var(Grad TD)
    """
    print(f"\n=================== Analyzing {env_name} ===================")
    is_mc = "mountaincar" in env_name.lower()
    cfg = base_cfg.copy()
    cfg["ENV_NAME"] = env_name
    cfg["TOTAL_TIMESTEPS"] = gt_updates
    cfg["NUM_TIMESTEPS"] = gt_updates
    cfg["ACTOR_LR"] = 0.001
    cfg["LAYER_NORM"] = True
    cfg["LIGHT_METRICS"] = True
    cfg["NETWORK_TYPE"] = "mlp" if is_mc else "cnn"
    cfg["USE_VISUAL_OBS"] = False if is_mc else True

    # 1. Train ground-truth policy
    print(f"Training ground_truth PPO for {gt_updates} updates...")
    t0 = time.time()
    train_fn = jax.jit(make_gt_train(cfg))
    rng = jax.random.PRNGKey(seed)
    gt_out = train_fn(rng)
    gt_time = time.time() - t0
    final_v_start = float(gt_out["metrics"]["V_start"][-1])
    print(f"Ground truth training finished in {gt_time:.1f}s | Final V_start: {final_v_start:.4f}")

    # 2. Extract policy from trained model
    train_state = gt_out["runner_state"][0]
    policy_params = train_state.params

    env, env_params = helpers.make_env(cfg)
    evaluator = helpers.initialize_evaluator(cfg, env, env_params)
    S = evaluator.obs_stack
    n_states = evaluator.num_states
    n_actions = evaluator.num_actions
    gamma = float(evaluator.gamma)
    P = np.array(evaluator.P)  # S x A x S' (or with terminal state S+1)
    R = np.array(evaluator.R)  # S x A x S' or S x A

    # Network inference to get policy
    net = train_state.apply_fn
    pi_dist, _ = net(policy_params, S)
    if hasattr(pi_dist, "probs"):
        pi = np.array(pi_dist.probs)
    else:
        action_basis = jnp.array(evaluator.directions, dtype=jnp.float32)
        log_probs = jax.vmap(lambda a: pi_dist.log_prob(a), in_axes=0, out_axes=-1)(action_basis)
        pi = np.array(jax.nn.softmax(log_probs, axis=-1))

    # Add terminal row if needed
    if pi.shape[0] == n_states:
        terminal_policy = np.ones([1, n_actions], dtype=pi.dtype) / n_actions
        pi_full = np.vstack([pi, terminal_policy])
    else:
        pi_full = pi

    # Compute stationary distribution mu
    mu_raw, _ = evaluator.compute_stationary_distribution_raw(pi[:n_states, :])
    mu = np.array(mu_raw)  # length n_states
    mu = np.maximum(mu, 0.0)
    if np.sum(mu) > 0:
        mu = mu / np.sum(mu)
    else:
        mu = np.ones(n_states) / n_states

    # Policy transition matrix P_pi
    # P has shape (n_states_total, n_actions, n_states_total)
    n_total = P.shape[0]
    P_pi = np.einsum("sa,sam->sm", pi_full[:n_total, :], P)

    # Reward under policy R_pi
    if R.ndim == 3:
        R_pi = np.einsum("sa,sam,sam->s", pi_full[:n_total, :], P, R)
    else:
        R_pi = np.einsum("sa,sa->s", pi_full[:n_total, :], R)

    # 3. Asymmetry: A = D(I - gamma P_pi) on continuing states
    mu_cont = mu[:n_states]
    D = np.diag(mu_cont)
    P_pi_cont = P_pi[:n_states, :n_states]

    A = D @ (np.eye(n_states) - gamma * P_pi_cont)
    S_mat = 0.5 * (A + A.T)
    K_mat = 0.5 * (A - A.T)

    eig_S = np.linalg.eigvalsh(S_mat)
    norm_S = float(np.max(np.abs(eig_S)))
    eig_K = np.linalg.eigvalsh(1j * K_mat)
    norm_K = float(np.max(np.abs(eig_K)))
    asymmetry_ratio = norm_K / (norm_S + 1e-12)
    frobenius_asymmetry = float(np.linalg.norm(K_mat, "fro") / (np.linalg.norm(S_mat, "fro") + 1e-12))

    # 4. Reward Density & Entropy
    R_cont = R_pi[:n_states]
    nonzero_mask = np.abs(R_cont) > 1e-5
    reward_density = float(np.sum(mu_cont * nonzero_mask))  # Probability of receiving reward
    reward_fraction_states = float(np.mean(nonzero_mask))
    expected_abs_reward = float(np.sum(mu_cont * np.abs(R_cont)))

    # State visitation entropy
    pos_mu = mu_cont[mu_cont > 1e-12]
    visitation_entropy = float(-np.sum(pos_mu * np.log(pos_mu)))

    # 5. Terminal Flux
    # Transitions going from continuing states into absorbing terminal state
    if n_total > n_states:
        # P has an explicit terminal state at index n_states
        p_to_term = P_pi[:n_states, n_states]
        terminal_flux = float(np.sum(mu_cont * p_to_term))
    else:
        # Check if rows of P_pi_cont sum to < 1.0 (sub-stochastic due to termination)
        row_sums = np.sum(P_pi_cont, axis=-1)
        leakage = np.maximum(0.0, 1.0 - row_sums)
        terminal_flux = float(np.sum(mu_cont * leakage))

    # 6. Sample Gradient Variance Penalty: Var(Grad E) / Var(Grad TD)
    # Using true value function V_pi
    V_pi = np.array(evaluator.compute_true_values_raw(pi_full))[:n_states]
    V_next = np.append(V_pi, 0.0) if n_total > n_states else V_pi

    # TD error for each transition (s, a, s'): R(s, a, s') + gamma * V(s') - V(s)
    if R.ndim == 3:
        delta_td = R[:n_states, :, :n_total] + gamma * V_next[None, None, :n_total] - V_pi[:n_states, None, None]
    else:
        delta_td = R[:n_states, :, None] + gamma * V_next[None, None, :n_total] - V_pi[:n_states, None, None]

    # Sample transition weight W(s, a, s') = mu(s) * pi(a|s) * P(s'|s,a)
    W = mu_cont[:, None, None] * pi[:n_states, :, None] * P[:n_states, :, :n_total]

    # TD sample gradient variance: Var(delta_td)
    var_grad_TD = float(np.sum(W * (delta_td ** 2)))

    # E-loss sample gradient:
    # For transitions to non-terminal s': g_E = (2 - gamma) * delta_td - gamma * delta_next
    # For transitions to terminal s': g_E = (2 - gamma) * delta_td
    # Average next-state error variance across continuing states:
    is_terminal_next = np.zeros(n_total, dtype=bool)
    if n_total > n_states:
        is_terminal_next[n_states:] = True

    # Weight of continuing vs terminal transitions:
    # Var(g_E) = (2 - gamma)^2 Var(delta) + (1 - is_term) * gamma^2 Var(delta_next)
    gamma_sq = gamma ** 2
    factor_E = np.where(is_terminal_next[None, None, :], (2.0 - gamma) ** 2, (2.0 - gamma) ** 2 + gamma_sq)
    var_grad_E = float(np.sum(W * factor_E * (delta_td ** 2)))
    grad_var_ratio = float(var_grad_E / (var_grad_TD + 1e-12))

    result = {
        "env_name": env_name,
        "num_states": n_states,
        "gamma": gamma,
        "norm_S": norm_S,
        "norm_K": norm_K,
        "asymmetry_ratio": asymmetry_ratio,
        "frobenius_asymmetry": frobenius_asymmetry,
        "reward_density": reward_density,
        "reward_fraction_states": reward_fraction_states,
        "expected_abs_reward": expected_abs_reward,
        "visitation_entropy": visitation_entropy,
        "terminal_flux": terminal_flux,
        "var_grad_TD": var_grad_TD,
        "var_grad_E": var_grad_E,
        "grad_var_ratio": grad_var_ratio,
        "final_v_start": final_v_start,
    }

    print(f"Results for {env_name}:")
    print(f"  ||K||_2 / ||S||_2:       {asymmetry_ratio:.4f}")
    print(f"  Reward Density:          {reward_density:.4f} (Entropy: {visitation_entropy:.4f})")
    print(f"  Terminal Flux:           {terminal_flux:.4f}")
    print(f"  Var(Grad E)/Var(Grad TD): {grad_var_ratio:.4f}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--gt_updates", type=int, default=80)
    parser.add_argument("--out_csv", type=str, default="results/env_properties.csv")
    args = parser.parse_args()

    import pandas as pd
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    existing_df = None
    existing_envs = set()
    if os.path.exists(args.out_csv):
        try:
            existing_df = pd.read_csv(args.out_csv)
            existing_envs = set(existing_df["env_name"].tolist())
            print(f"Found {len(existing_envs)} already processed environments in {args.out_csv}")
        except Exception:
            existing_df = None

    envs_to_run = [args.env_name] if args.env_name else ALL_ENVS
    all_rows = existing_df.to_dict("records") if existing_df is not None else []

    for env in envs_to_run:
        if env in existing_envs and args.env_name is None:
            print(f"Skipping {env} (already computed).")
            continue

        res = analyze_env(env, gt_updates=args.gt_updates)
        # Update or append
        all_rows = [r for r in all_rows if r["env_name"] != env] + [res]
        # Save incrementally
        df = pd.DataFrame(all_rows)
        df.to_csv(args.out_csv, index=False)
        print(f"Progress saved to: {args.out_csv}")

    print(f"\nAll environment properties saved to: {args.out_csv}")
