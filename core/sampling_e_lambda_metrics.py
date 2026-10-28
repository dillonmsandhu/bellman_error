"""
core/sampling_e_lambda_metrics.py

Computes exact vs sampled E(lambda) objective metrics and critic parameter gradients:
1. Exact Matrix E(lambda): e^T S_lambda e, where S_lambda = 0.5 * (A_lambda + A_lambda.T),
   and A_lambda = D (I - gamma P) (I - gamma lambda P)^{-1}.
2. Naive Sampling E(lambda) (analogous to Oct 8):
   L = w_mag * E[e^2] + w_dir * E[(e_t - e_j)^2] + w_start * E_{mu_0}[e_t^2]
   Differentiates both source state s_t and geometric jump destination s_j.
3. Aux Version E(lambda) (v_aux debiasing):
   L = w_mag * E[e^2] + w_dir * E[(e_t - sg(e_j))^2 + (sg(e_tilde_t) - e_tilde_j)^2] + w_start * E_{mu_0}[e_t^2]
4. Forward Only E(lambda):
   L = w_mag * E[e^2] + w_dir * E[(e_t - sg(e_j))^2] + w_start * E_{mu_0}[e_t^2]
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional

from core.gradient_tracking import extract_critic_flat_grads, cos_similarity


def compute_sampling_e_lambda_metrics(
    evaluator,
    network,
    params,
    mu: jnp.ndarray,
    V_true: jnp.ndarray,
    P_pi: jnp.ndarray,
    S_mat: jnp.ndarray,
    gamma: float,
    lmbda: float,
    rng: jax.Array,
    num_trajectories: int = 128,
    num_steps: int = 128,
    aux_network: Optional[Any] = None,
    aux_train_state: Optional[Any] = None,
    aux_steps: int = 32,
    aux_target_mode: str = "true_value",
    runway_length: int = 16,
) -> Tuple[Dict[str, jnp.ndarray], Optional[Any]]:
    """
    Computes exact, expected, and sampled E(lambda) metrics and gradients across:
    - Naive sampling (Oct 8 analog: differentiates both ends of geometric jump)
    - Aux version (debiased surrogate via v_aux)
    - Forward-only version (drops backward branch)
    """
    S_states = evaluator.obs_stack
    N = evaluator.num_states
    start_idx = evaluator.start_idx

    # Forward value prediction helper
    def get_e_full(p):
        v = network.apply(p, S_states, method=network.value).squeeze()
        v_full = jnp.append(v, 0.0)
        return V_true - v_full

    # 1. Rollout sampling from mu: B parallel trajectories of T steps
    rng_init, rng_steps, rng_jump = jax.random.split(rng, 3)
    mu_trans = mu[:N] / jnp.maximum(jnp.sum(mu[:N]), 1e-12)
    s0 = jax.random.choice(rng_init, N, shape=(num_trajectories,), p=mu_trans)

    # Current policy on discrete state stack
    pi_probs = network.apply(params, S_states, method=network.policy).probs
    pi_full = jnp.vstack([pi_probs, jnp.ones([1, pi_probs.shape[-1]]) / pi_probs.shape[-1]])
    P_env = evaluator.P
    R_env = evaluator.R

    def step_fn(carry_s, step_rng):
        rng_a, rng_s = jax.random.split(step_rng)
        step_rngs_a = jax.random.split(rng_a, num_trajectories)
        step_rngs_s = jax.random.split(rng_s, num_trajectories)
        logits_pi = jnp.log(jnp.maximum(pi_full[carry_s], 1e-12))
        actions = jax.vmap(jax.random.categorical)(step_rngs_a, logits_pi)
        P_sa = P_env[carry_s, actions]
        logits_p = jnp.log(jnp.maximum(P_sa, 1e-12))
        next_s_raw = jax.vmap(jax.random.categorical)(step_rngs_s, logits_p)
        rewards = R_env[carry_s, actions, next_s_raw]
        dones = (next_s_raw == N)
        next_s_carry = jnp.where(dones, start_idx, next_s_raw)
        return next_s_carry, (carry_s, next_s_raw, rewards, dones)

    step_rngs = jax.random.split(rng_steps, num_steps)
    final_s, (traj_s, traj_next_s, traj_r, traj_done) = jax.lax.scan(step_fn, s0, step_rngs)

    # 1b. Compute empirical returns G_t along each trajectory
    v_cur = network.apply(params, S_states, method=network.value).squeeze()
    v_full_cur = jnp.append(v_cur, 0.0)
    v_final = v_full_cur[final_s]

    def rev_scan_fn(g_next, step_data):
        r_t, d_t = step_data
        g_cur = r_t + gamma * jnp.where(d_t, 0.0, g_next)
        return g_cur, g_cur

    _, returns = jax.lax.scan(rev_scan_fn, v_final, (traj_r, traj_done), reverse=True)
    returns_sg = jax.lax.stop_gradient(returns)

    # 1c. Update auxiliary network state for v_aux if provided
    new_aux_train_state = aux_train_state
    if aux_train_state is not None and aux_network is not None:
        if aux_target_mode == "true_value":
            def aux_step_fn(ts, _):
                def loss_fn(p_aux):
                    v_aux_pred = aux_network.apply(p_aux, S_states).squeeze()
                    return jnp.mean((V_true[:N] - v_aux_pred) ** 2)
                grads = jax.grad(loss_fn)(ts.params)
                return ts.apply_gradients(grads=grads), None

            new_aux_train_state, _ = jax.lax.scan(aux_step_fn, aux_train_state, None, length=aux_steps)
        else:
            def aux_step_fn(ts, _):
                def loss_fn(p_aux):
                    v_aux_pred = aux_network.apply(p_aux, S_states).squeeze()
                    v_aux_full = jnp.append(v_aux_pred, 0.0)
                    return jnp.mean((returns_sg - v_aux_full[traj_s]) ** 2)
                grads = jax.grad(loss_fn)(ts.params)
                return ts.apply_gradients(grads=grads), None

            new_aux_train_state, _ = jax.lax.scan(aux_step_fn, aux_train_state, None, length=aux_steps)

        v_aux = jnp.append(aux_network.apply(new_aux_train_state.params, S_states).squeeze(), 0.0)
    else:
        v_aux = V_true

    v_aux_sg = jax.lax.stop_gradient(v_aux)

    # 2. Geometric Jump Lookaheads for E(lambda)
    T, B = num_steps, num_trajectories
    T_start = min(T, max(1, T - runway_length))
    gl = gamma * lmbda
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)

    u = jax.random.uniform(rng_jump, shape=(T_start, B))
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T_start, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    m = jnp.maximum(jumps, 0) + 1
    t_arr = jnp.arange(T_start)[:, None]
    j = jnp.minimum(t_arr + m, T)

    # Check terminal boundaries in [t, j - 1]
    # In exact_E environments with no artificial timeout, done is true_terminal
    true_terminals = traj_done
    term_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(true_terminals.astype(jnp.int32), axis=0)],
        axis=0,
    )
    hit_true_terminal = (jnp.take_along_axis(term_cumsum, j, axis=0) - term_cumsum[:T_start]) > 0
    hit_timeout = jnp.zeros_like(hit_true_terminal, dtype=bool)

    # Track steps since start for geometrically decaying reset grounding
    def scan_steps(k, term):
        k_next = jnp.where(term, 0, k + 1)
        return k_next, k

    init_k = jnp.full((B,), 10_000, dtype=jnp.int32)
    _, all_steps = jax.lax.scan(scan_steps, init_k, true_terminals)

    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    start_trace = jnp.where(
        gl < 1e-6,
        jnp.where(all_steps[:T_start] == 0, 1.0, 0.0),
        safe_gl ** all_steps[:T_start],
    )
    start_weight = 0.25 * tilde_gamma * start_trace

    # Form extended arrays for destination lookups at step T
    traj_s_ext = jnp.concatenate([traj_s, final_s[None]], axis=0)
    target_ext = jnp.concatenate([returns_sg, v_final[None]], axis=0)

    dest_s = jnp.take_along_axis(traj_s_ext, j, axis=0)
    dest_target = jnp.take_along_axis(target_ext, j, axis=0)

    # Prefactors
    w_mag = 0.5 * (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    w_dir = 0.5 * tilde_gamma

    # 3. Exact Matrix Loss & Gradient
    def loss_exact(p):
        e_cur = get_e_full(p)
        return 0.5 * (e_cur.T @ S_mat @ e_cur)

    l_exact = loss_exact(params)
    g_exact = extract_critic_flat_grads(jax.grad(loss_exact)(params))
    norm_g_exact = jnp.linalg.norm(g_exact)

    # 4. Trajectory-Level Gradient Analysis & Vectorization across B Trajectories
    # Transpose to (B, T_start) for vmap over trajectories
    s_trajs = jnp.transpose(traj_s[:T_start], (1, 0))
    dest_s_trajs = jnp.transpose(dest_s, (1, 0))
    ret_trajs = jnp.transpose(returns_sg[:T_start], (1, 0))
    dest_ret_trajs = jnp.transpose(dest_target, (1, 0))
    hit_term_trajs = jnp.transpose(hit_true_terminal, (1, 0))
    hit_time_trajs = jnp.transpose(hit_timeout, (1, 0))
    w_start_trajs = jnp.transpose(start_weight, (1, 0))

    sg = jax.lax.stop_gradient

    # Single trajectory gradient functions:
    # 4a. Naive Sampling (Oct 8 analog: differentiates both ends of jump)
    def single_traj_grad_naive(s_b, dest_s_b, ret_b, dest_ret_b, hit_term_b, hit_time_b, w_start_b):
        def l_fn(p):
            v = jnp.append(network.apply(p, S_states, method=network.value).squeeze(), 0.0)
            es = ret_b - v[s_b]
            ej = jnp.where(hit_term_b, 0.0, dest_ret_b - v[dest_s_b])

            diff_sq = jnp.where(
                hit_term_b,
                es ** 2,
                jnp.where(hit_time_b, 0.0, (es - ej) ** 2),
            )
            mag = w_mag * jnp.mean(es ** 2)
            dir_loss = w_dir * jnp.mean(diff_sq)
            start = jnp.mean(w_start_b * (es ** 2))
            return mag + dir_loss + start

        return extract_critic_flat_grads(jax.grad(l_fn)(params))

    # 4b. Aux Version (debiased surrogate using v_aux)
    def single_traj_grad_aux(s_b, dest_s_b, ret_b, dest_ret_b, hit_term_b, hit_time_b, w_start_b):
        def l_fn(p):
            v = jnp.append(network.apply(p, S_states, method=network.value).squeeze(), 0.0)
            es = ret_b - v[s_b]
            ej = jnp.where(hit_term_b, 0.0, dest_ret_b - v[dest_s_b])

            e_tilde_s = v_aux_sg[s_b] - v[s_b]
            e_tilde_j = jnp.where(hit_term_b, 0.0, v_aux_sg[dest_s_b] - v[dest_s_b])

            diff_sq = jnp.where(
                hit_term_b,
                es ** 2,
                jnp.where(
                    hit_time_b,
                    0.0,
                    (es - sg(ej)) ** 2 + (sg(e_tilde_s) - e_tilde_j) ** 2,
                ),
            )
            mag = w_mag * jnp.mean(es ** 2)
            dir_loss = w_dir * jnp.mean(diff_sq)
            start = jnp.mean(w_start_b * (es ** 2))
            return mag + dir_loss + start

        return extract_critic_flat_grads(jax.grad(l_fn)(params))

    # 4c. Forward Only (drops backward branch)
    def single_traj_grad_fwd(s_b, dest_s_b, ret_b, dest_ret_b, hit_term_b, hit_time_b, w_start_b):
        def l_fn(p):
            v = jnp.append(network.apply(p, S_states, method=network.value).squeeze(), 0.0)
            es = ret_b - v[s_b]
            ej = jnp.where(hit_term_b, 0.0, dest_ret_b - v[dest_s_b])

            diff_sq = jnp.where(
                hit_term_b,
                es ** 2,
                jnp.where(hit_time_b, 0.0, (es - sg(ej)) ** 2),
            )
            mag = w_mag * jnp.mean(es ** 2)
            dir_loss = w_dir * jnp.mean(diff_sq)
            start = jnp.mean(w_start_b * (es ** 2))
            return mag + dir_loss + start

        return extract_critic_flat_grads(jax.grad(l_fn)(params))

    # Vectorized gradient computation across all B trajectories
    args_trajs = (s_trajs, dest_s_trajs, ret_trajs, dest_ret_trajs, hit_term_trajs, hit_time_trajs, w_start_trajs)
    G_naive = jax.vmap(single_traj_grad_naive)(*args_trajs)
    G_aux = jax.vmap(single_traj_grad_aux)(*args_trajs)
    G_fwd = jax.vmap(single_traj_grad_fwd)(*args_trajs)

    # Total average gradient across the full rollout batch (B trajectories)
    g_bar_naive = jnp.mean(G_naive, axis=0)
    g_bar_aux = jnp.mean(G_aux, axis=0)
    g_bar_fwd = jnp.mean(G_fwd, axis=0)

    # Full batch cosine similarities to exact gradient g*
    rho_s_naive = cos_similarity(g_bar_naive, g_exact)
    rho_s_aux = cos_similarity(g_bar_aux, g_exact)
    rho_s_fwd = cos_similarity(g_bar_fwd, g_exact)

    # Trajectory gradient variance (spread of trajectories): (1/B) sum ||g_b - g_bar||^2
    var_traj_naive = jnp.mean(jnp.sum((G_naive - g_bar_naive[None, :]) ** 2, axis=1))
    var_traj_aux = jnp.mean(jnp.sum((G_aux - g_bar_aux[None, :]) ** 2, axis=1))
    var_traj_fwd = jnp.mean(jnp.sum((G_fwd - g_bar_fwd[None, :]) ** 2, axis=1))

    # Variance of the mean gradient estimator: Var(g_bar) = Var_traj / B
    var_naive = var_traj_naive / float(num_trajectories)
    var_aux = var_traj_aux / float(num_trajectories)
    var_fwd = var_traj_fwd / float(num_trajectories)

    # Total MSE of average gradient to exact g*
    mse_naive = jnp.sum((g_bar_naive - g_exact) ** 2)
    mse_aux = jnp.sum((g_bar_aux - g_exact) ** 2)
    mse_fwd = jnp.sum((g_bar_fwd - g_exact) ** 2)

    norm_naive = jnp.linalg.norm(g_bar_naive)
    norm_aux = jnp.linalg.norm(g_bar_aux)
    norm_fwd = jnp.linalg.norm(g_bar_fwd)

    # Sample loss values across entire batch
    es_all = returns_sg[:T_start] - v_full_cur[traj_s[:T_start]]
    ej_all = jnp.where(hit_true_terminal, 0.0, dest_target - v_full_cur[dest_s])
    e_tilde_s_all = v_aux_sg[traj_s[:T_start]] - v_full_cur[traj_s[:T_start]]
    e_tilde_j_all = jnp.where(hit_true_terminal, 0.0, v_aux_sg[dest_s] - v_full_cur[dest_s])

    mag_all = w_mag * jnp.mean(es_all ** 2)
    start_all = jnp.mean(start_weight * (es_all ** 2))

    dir_naive_all = w_dir * jnp.mean(jnp.where(hit_true_terminal, es_all ** 2, (es_all - ej_all) ** 2))
    dir_aux_all = w_dir * jnp.mean(
        jnp.where(
            hit_true_terminal,
            es_all ** 2,
            (es_all - sg(ej_all)) ** 2 + (sg(e_tilde_s_all) - e_tilde_j_all) ** 2,
        )
    )
    dir_fwd_all = w_dir * jnp.mean(jnp.where(hit_true_terminal, es_all ** 2, (es_all - sg(ej_all)) ** 2))

    l_samp_naive = mag_all + dir_naive_all + start_all
    l_samp_aux = mag_all + dir_aux_all + start_all
    l_samp_fwd = mag_all + dir_fwd_all + start_all

    metrics = {
        "E_exact": l_exact,
        "E_samp_naive": l_samp_naive,
        "E_samp_aux": l_samp_aux,
        "E_samp_fwd": l_samp_fwd,
        "norm_g_exact": norm_g_exact,
        # Full-Batch Average Gradient Cosine Similarities to Exact E(lambda)
        "rho_samp_naive_exact": rho_s_naive,
        "rho_samp_aux_exact": rho_s_aux,
        "rho_samp_fwd_exact": rho_s_fwd,
        # Trajectory gradient variance (spread of trajectories)
        "var_traj_samp_naive": var_traj_naive,
        "var_traj_samp_aux": var_traj_aux,
        "var_traj_samp_fwd": var_traj_fwd,
        # Variance of the average gradient estimator
        "var_samp_naive": var_naive,
        "var_samp_aux": var_aux,
        "var_samp_fwd": var_fwd,
        # Total MSE of average gradient to exact g*
        "mse_samp_naive": mse_naive,
        "mse_samp_aux": mse_aux,
        "mse_samp_fwd": mse_fwd,
        # Gradient norms of average gradients
        "norm_g_samp_naive": norm_naive,
        "norm_g_samp_aux": norm_aux,
        "norm_g_samp_fwd": norm_fwd,
        # Norm ratios
        "norm_ratio_naive": norm_naive / jnp.maximum(1e-8, norm_g_exact),
        "norm_ratio_aux": norm_aux / jnp.maximum(1e-8, norm_g_exact),
        "norm_ratio_fwd": norm_fwd / jnp.maximum(1e-8, norm_g_exact),
        # Auxiliary value network accuracy to V_true
        "mse_v_aux_target": jnp.mean((V_true - v_aux) ** 2),
    }

    if aux_train_state is not None:
        return metrics, new_aux_train_state
    return metrics, None
