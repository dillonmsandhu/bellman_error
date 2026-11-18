"""
Experiment 1: 1D Chain Random-Walk MRP (TD(0) vs. E(0)).

Sweeps:
  - Rows (transition drift p_fwd): 1.00, 0.75, 0.50 (with p_bwd = 1 - p_fwd)
  - Columns (network on 1-hot state):
      1. 1-Hot + MLP               (core.networks.MLP + linear value head)
      2. 1-Hot + CNN               (Conv1D k=5)
      3. 1-Hot + CNN (AvgPool)     (AvgPool w=7 + Conv1D k=5)

Saves results to investigation/figures/chain_demo_1.{png,pdf}.
"""

from __future__ import annotations
import os
import sys
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.networks import MLP


def build_chain_mrp(N: int = 32, gamma: float = 0.99, p_fwd: float = 1.0, ring: bool = False):
    """
    1D chain (or closed ring) with states 0..N-1:
      - Move forward (+1) with probability p_fwd, backward (-1) with probability (1 - p_fwd).
      - If ring=False:
          - Left boundary (s=0): backward steps stay at 0.
          - Right boundary (s=N-1): forward steps exit to terminal (value 0) with reward +1.0,
            resetting to s=0 in the continuing occupancy distribution mu.
      - If ring=True:
          - Closed ring mod N (continuing, no terminal exit): N-1 -> 0 with p_fwd (reward +1.0)
            and 0 -> N-1 with (1 - p_fwd).
    """
    p_bwd = 1.0 - p_fwd
    P_ep = np.zeros((N, N), dtype=np.float64)
    if ring:
        for i in range(N):
            P_ep[i, (i + 1) % N] += p_fwd
            P_ep[i, (i - 1) % N] += p_bwd
        P_c = P_ep.copy()
    else:
        for i in range(N):
            if i > 0:
                P_ep[i, i - 1] += p_bwd
            else:
                P_ep[i, 0] += p_bwd
            if i < N - 1:
                P_ep[i, i + 1] += p_fwd

        P_c = P_ep.copy()
        P_c[N - 1, 0] += p_fwd

    eigvals, eigvecs = np.linalg.eig(P_c.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    mu = np.real(eigvecs[:, idx])
    if mu.sum() < 0:
        mu = -mu
    mu = np.maximum(mu, 0.0)
    mu /= mu.sum()
    D = np.diag(mu)

    R_pi = np.zeros(N, dtype=np.float64)
    R_pi[N - 1] = p_fwd * 1.0

    V = np.linalg.solve(np.eye(N) - gamma * P_ep, R_pi)
    A = D @ (np.eye(N) - gamma * P_ep)
    S = 0.5 * (A + A.T)
    return {
        "N": N,
        "gamma": gamma,
        "p_fwd": p_fwd,
        "P_ep": P_ep,
        "R_pi": R_pi,
        "mu": mu,
        "V": V,
        "S": S,
    }


class MLPValueNet(nn.Module):
    """Uses core.networks.MLP backbone with a scalar value head."""

    @nn.compact
    def __call__(self, x):
        feats = MLP(norm_type="layer_norm", final_hidden_dim=32, hidden_dim=64)(x)
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(feats).squeeze(-1)


class ChainCNN(nn.Module):
    """1D CNN over 1-hot state input, with optional spatial average pooling."""
    use_avg_pool: bool = False
    pool_width: int = 7
    kernel_size: int = 5
    features: int = 16
    final_dim: int = 32

    @nn.compact
    def __call__(self, x):
        if self.use_avg_pool:
            x = nn.avg_pool(x, window_shape=(self.pool_width,), strides=(1,), padding="SAME")
            x = nn.Conv(self.features, kernel_size=(self.kernel_size,), padding="SAME")(x)
            x = nn.tanh(x)
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(self.final_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(x)
            x = nn.tanh(x)
        else:
            x = nn.Conv(
                self.features,
                kernel_size=(self.kernel_size,),
                padding="SAME",
                kernel_init=nn.initializers.he_normal(),
            )(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(self.final_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x).squeeze(-1)


def make_model(col: str, N: int):
    if col == "onehot_mlp":
        return jnp.eye(N, dtype=jnp.float32), MLPValueNet()
    elif col == "onehot_cnn":
        return jnp.eye(N, dtype=jnp.float32)[:, :, None], ChainCNN(use_avg_pool=False)
    elif col == "onehot_cnn_pool":
        return jnp.eye(N, dtype=jnp.float32)[:, :, None], ChainCNN(use_avg_pool=True)
    raise ValueError(f"Unknown model column: {col}")


def compute_grad_corr(net, params, obs, mrp):
    """
    Centered next-state gradient cosine similarity rho(grad v(s), grad v(s')):
        Tr(J_c^T D P J_c) / Tr(J_c^T D J_c),
    where J_c centers the value gradients grad_theta v(s) across mu.
    """
    def single_v(p, x):
        return net.apply(p, x[None, ...])[0]

    J_tree = jax.vmap(jax.jacrev(single_v), in_axes=(None, 0))(params, obs)
    J = jnp.concatenate(
        [jnp.reshape(x, (obs.shape[0], -1)) for x in jax.tree_util.tree_leaves(J_tree)],
        axis=1,
    )
    mu = jnp.asarray(mrp["mu"], dtype=jnp.float32)[:, None]
    P = jnp.asarray(mrp["P_ep"], dtype=jnp.float32)
    J_c = J - jnp.sum(mu * J, axis=0, keepdims=True)
    D_Jc = mu * J_c
    num = jnp.sum(D_Jc * (P @ J_c))
    den = jnp.maximum(jnp.sum(D_Jc * J_c), 1e-12)
    return num / den


def make_runner(mrp, col: str, algo: str, steps: int, eval_every: int = 50):
    obs, net = make_model(col, mrp["N"])
    P_ep = jnp.asarray(mrp["P_ep"], dtype=jnp.float32)
    R_pi = jnp.asarray(mrp["R_pi"], dtype=jnp.float32)
    mu = jnp.asarray(mrp["mu"], dtype=jnp.float32)
    V = jnp.asarray(mrp["V"], dtype=jnp.float32)
    S = jnp.asarray(mrp["S"], dtype=jnp.float32)
    gamma = float(mrp["gamma"])
    V_norm = jnp.sum(mu * (V ** 2))
    tx = optax.inject_hyperparams(optax.adam)(learning_rate=1e-3)
    n_chunks = steps // eval_every

    def loss_fn(p):
        v = net.apply(p, obs)
        e = V - v
        if algo == "e0":
            return 0.5 * e @ S @ e
        elif algo == "td0":
            target = jax.lax.stop_gradient(R_pi + gamma * P_ep @ v)
            return 0.5 * jnp.sum(mu * (v - target) ** 2)
        raise ValueError(f"Unknown algo: {algo}")

    grad_fn = jax.value_and_grad(loss_fn)

    def run_seed(seed_key, lr):
        params = net.init(seed_key, obs)
        opt_state = tx.init(params)
        opt_state.hyperparams["learning_rate"] = lr

        def step_fn(carry, _):
            p, os_ = carry
            v = net.apply(p, obs)
            rel_mse = jnp.sum(mu * (V - v) ** 2) / V_norm
            _, g = grad_fn(p)
            updates, os_ = tx.update(g, os_, p)
            p = optax.apply_updates(p, updates)
            return (p, os_), rel_mse

        def chunk_fn(carry, _):
            p, os_ = carry
            gc = compute_grad_corr(net, p, obs, mrp)
            (p, os_), mses_chunk = jax.lax.scan(step_fn, (p, os_), None, length=eval_every)
            return (p, os_), (mses_chunk, gc)

        (p_end, _), (mse_chunks, gc_hist) = jax.lax.scan(chunk_fn, (params, opt_state), None, length=n_chunks)
        gc_final = compute_grad_corr(net, p_end, obs, mrp)
        rel_mses = mse_chunks.reshape((-1,))
        grad_corrs = jnp.concatenate([gc_hist, gc_final[None]], axis=0)
        return rel_mses, grad_corrs

    return jax.jit(jax.vmap(run_seed, in_axes=(0, None)))


def main():
    t0 = time.time()
    N = 32
    gamma = 0.99
    steps = 1500
    seeds = jnp.stack([jax.random.PRNGKey(s) for s in range(5)])
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    row_specs = [
        ("p_fwd = 1.00", 1.0, False),
        ("p_fwd = 0.75", 0.75, False),
        ("p_fwd = 0.50", 0.5, False),
        ("p_fwd = 0.75 (Ring)", 0.75, True),
    ]
    cols = ["onehot_mlp", "onehot_cnn", "onehot_cnn_pool"]
    algos = ["td0", "e0"]

    eval_every = 50
    results = {}
    for row_label, p_fwd, ring in row_specs:
        mrp = build_chain_mrp(N=N, gamma=gamma, p_fwd=p_fwd, ring=ring)
        print(f"\n--- {row_label} ---")
        for col in cols:
            for algo in algos:
                run_vmap = make_runner(mrp, col, algo, steps, eval_every=eval_every)
                best_auc = float("inf")
                best_entry = None
                for lr in lrs:
                    curves, gcs = run_vmap(seeds, lr)
                    curves = np.asarray(curves)
                    gcs = np.asarray(gcs)
                    aucs = curves.mean(axis=1)
                    mean_auc = float(np.mean(aucs))
                    if np.isfinite(mean_auc) and mean_auc < best_auc:
                        best_auc = mean_auc
                        best_entry = {
                            "lr": lr,
                            "auc_mean": mean_auc,
                            "auc_sem": float(np.std(aucs) / np.sqrt(len(aucs))),
                            "final_mean": float(np.mean(curves[:, -1])),
                            "grad_corr": gcs,
                        }
                results[(row_label, col, algo)] = best_entry
                e = best_entry
                print(
                    f"  {col:16s} | {algo:3s} | lr={e['lr']:<6.4g} | "
                    f"AUC={e['auc_mean']:.4f}±{e['auc_sem']:.4f} Fin={e['final_mean']:.4f} | "
                    f"rho(0)={e['grad_corr'][:, 0].mean():+.3f}->rho(end)={e['grad_corr'][:, -1].mean():+.3f}"
                )

    fig_dir = os.path.join(REPO_ROOT, "investigation", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(fig_dir, "chain_demo_1_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    row_labels = [r[0] for r in row_specs]
    col_labels = ["1-Hot + MLP", "1-Hot + CNN", "1-Hot + CNN (AvgPool)"]

    mat = np.zeros((len(row_specs), len(cols)))
    for i, row_label in enumerate(row_labels):
        for j, c in enumerate(cols):
            mat[i, j] = results[(row_label, c, "td0")]["auc_mean"] / results[(row_label, c, "e0")]["auc_mean"]

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 13})
    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    vmax = min(max(4.0, float(np.ceil(mat.max()))), 15.0)
    im = ax.imshow(mat, cmap="YlGnBu", vmin=1.0, vmax=vmax, aspect="auto")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(col_labels, fontweight="bold", fontsize=11)
    ax.set_yticks(list(range(len(row_labels))))
    ax.set_yticklabels(row_labels, fontweight="bold", fontsize=11)

    thresh = 1.0 + 0.55 * (vmax - 1.0)
    for i, row_label in enumerate(row_labels):
        for j, c in enumerate(cols):
            td_e = results[(row_label, c, "td0")]
            e0_e = results[(row_label, c, "e0")]
            ratio = td_e["auc_mean"] / e0_e["auc_mean"]
            color = "white" if ratio > thresh else "#111111"
            txt = f"{ratio:.2f}×\nTD: {td_e['auc_mean']:.4f} | E: {e0_e['auc_mean']:.4f}"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10.5, fontweight="bold", linespacing=1.45)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("AUC(TD) / AUC(E)", fontweight="bold")
    ax.set_title("1D Random Walk Chain MRP (N=32, γ=0.99)", fontweight="bold", pad=12)
    ax.set_xlabel("AUC Ratio: AUC(TD) / AUC(E) (lower is better for TD)", fontweight="bold", labelpad=12)
    plt.tight_layout()

    png_path = os.path.join(fig_dir, "chain_demo_1.png")
    pdf_path = os.path.join(fig_dir, "chain_demo_1.pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    # Plot 2: Transition Gradient Correlation (M_2) over update steps (p_fwd = 0.50 vs 0.75)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    xs = np.arange(0, steps + 1, eval_every)
    net_colors = {
        "onehot_mlp": "#1f77b4",
        "onehot_cnn": "#e67e22",
        "onehot_cnn_pool": "#7b2cbf",
    }
    net_names = {
        "onehot_mlp": "1-Hot + MLP",
        "onehot_cnn": "1-Hot + CNN",
        "onehot_cnn_pool": "1-Hot + CNN (AvgPool)",
    }
    p_styles = {"p_fwd = 0.75": "-", "p_fwd = 0.50": "--"}

    for ax_i, (algo, algo_title) in enumerate([("td0", "TD(0)"), ("e0", "E(0)")]):
        ax2 = axes2[ax_i]
        for col in cols:
            for r_lbl, pf_val in [("p_fwd = 0.75", 0.75), ("p_fwd = 0.50", 0.50)]:
                gcs = results[(r_lbl, col, algo)]["grad_corr"]
                mean_gc = gcs.mean(axis=0)
                sem_gc = gcs.std(axis=0) / np.sqrt(gcs.shape[0])
                lbl = f"{net_names[col]} (p={pf_val:.2f})"
                ax2.plot(xs, mean_gc, color=net_colors[col], ls=p_styles[r_lbl], lw=2.0, label=lbl)
                ax2.fill_between(xs, mean_gc - sem_gc, mean_gc + sem_gc, color=net_colors[col], alpha=0.12)
        ax2.axhline(0.0, color="#666666", ls=":", lw=1.0)
        ax2.set_title(f"{algo_title}: $\\rho(\\nabla v(s), \\nabla v(s'))$ Over Training", fontweight="bold")
        ax2.set_xlabel("Gradient Steps (Adam)")
        if ax_i == 0:
            ax2.set_ylabel("Centered $\\rho(\\nabla v(s), \\nabla v(s'))$")
        ax2.set_ylim(-0.2, 1.05)
        ax2.grid(True, alpha=0.25, lw=0.6)
        ax2.legend(loc="lower right", fontsize=8.2, ncol=1, framealpha=0.9)

    fig2.suptitle("Next-State Gradient Similarity $\\rho(\\nabla v(s), \\nabla v(s'))$ Over Update Steps (p_fwd = 0.50 vs. 0.75)", fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    m2_png = os.path.join(fig_dir, "chain_demo_1_m2.png")
    m2_pdf = os.path.join(fig_dir, "chain_demo_1_m2.pdf")
    fig2.savefig(m2_png, dpi=220)
    fig2.savefig(m2_pdf)
    plt.close(fig2)
    print(f"\nSaved {png_path} and {m2_png} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
