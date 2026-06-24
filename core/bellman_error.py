# bellman_error.py
# Metrics:
# Errors: Value Error, Bellman Error, and Projected Bellman Error
# Greedy Policy Alignmnet: How many states agree for what action to take?
# Feature Space Quality: Effective Dim, Value Projection Angle, Value PCA
from core.imports import *
import distrax
# from sklearn.decomposition import PCA

ε =  0.0

# Four Rooms has 104 states + an "invisible" terminal state.

def Bellman_Residual_Exact(D,Φ,P_π, R_π, γ):
    X = (Φ - γ * P_π @ Φ)
    reg = ε * jnp.eye(Φ.shape[-1]) / 100 # too large compared to LSTD...
    # w_br = jnp.linalg.solve(X.T @ D @ X + reg, X.T @ D @ R_π)
    w_br = jnp.linalg.pinv(X.T @ D @ X + reg) @ X.T @ D @ R_π
    V_br = Φ @ w_br
    return V_br, w_br

def LSTD_Exact(D, Φ, P_π, R_π, γ):
    # LSTD
    X = (Φ - γ * P_π @ Φ)
    A = Φ.T @ D @ X
    b = Φ.T @ D @ R_π
    reg =  ε * jnp.eye(Φ.shape[-1])
    w_lstd = jnp.linalg.pinv(A+reg) @ b
    # w_lstd = jnp.linalg.solve(A + reg, b)
    V_lstd = Φ @ w_lstd
    return V_lstd, w_lstd

def LeastSquaresValue(D,Φ,V_true):
    # reg = ε * jnp.eye(Φ.shape[-1])
    # reg_matrix = reg_matrix.at[-1, -1].set(0.0)  # Do NOT regularize the bias column
    # w_vr = jnp.linalg.pinv(Φ.T @ D @ Φ + reg) @ Φ.T @ D @ V_true
    # w_vr = jnp.linalg.solve(Φ.T @ D @ Φ + reg, Φ.T @ D @ V_true)
    
    D_sqrt = jnp.sqrt(D)
    w_vr, _, _, _ = jnp.linalg.lstsq(D_sqrt @ Φ, D_sqrt @ V_true)
    V_vr = Φ @ w_vr
    return V_vr, w_vr
    # D_sqrt = jnp.sqrt(D)
    # w_vr, _, _, _ = jnp.linalg.lstsq(D_sqrt @ Φ, D_sqrt @ V_true)

def get_error_vectors(V, v_pred, D, R_π, P_π, γ, Φ):
    Π_φ = Φ @ jnp.linalg.pinv(Φ.T @ D @ Φ) @ Φ.T @ D # projection matrix
    T = lambda v: R_π + γ * P_π @ v
    BE = T(v_pred) - v_pred
    PBE = Π_φ @ BE
    VE = V - v_pred
    Bellman_Orthogonal_Portion = T(v_pred) - Π_φ @ T(v_pred) # Tv - ΠTv
    return {'BE': BE, 'PBE': PBE, 'VE': VE, "Bellman_Orthogonal_Portion": Bellman_Orthogonal_Portion}

def compute_greedy_policy(P, R_π_s, γ, v):
    """
    Compute the greedy policy according to the value estimate v.
    
    Args:
        P: Transition dynamics tensor of shape (S, A, S)
        R: Extrinsic reward matrix of shape (S, A)
    """
    R_shifted = jnp.einsum("sam,m->sa", P, R_π_s)
    expected_v = jnp.einsum("sam,m->sa", P, v)
    Qs = R_shifted + γ * expected_v
    return jnp.argmax(Qs, axis=-1)

# def weighted_PCA(D, Φ):
    # sqrt_d_weights = jnp.sqrt(jnp.diag(D))
    # Weighted_Phi = sqrt_d_weights[:, None] * Φ
    # pca = PCA(n_components=2)
    # # Ignore terminal state
    # phi_2d = pca.fit_transform(Weighted_Phi[:-1, :])
    # return phi_2d # N x 2

def weighted_PCA(D, Φ):
    # 1. Weight the features
    sqrt_d_weights = jnp.sqrt(jnp.diag(D))
    Weighted_Phi = sqrt_d_weights[:, None] * Φ
    
    # Ignore terminal state
    X = Weighted_Phi[:-1, :]
    
    # 2. Pure JAX PCA
    # Center the data
    X_centered = X - jnp.mean(X, axis=0)
    
    # Compute SVD
    U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)
    
    # Project down to the first 2 principal components
    # (U * S is mathematically equivalent to X_centered @ V, but faster since we have U & S)
    phi_2d = U[:, :2] * S[:2]
    
    return phi_2d # N x 2

def get_capacity_angle(V_true, V_vr, D):
    "cosine(theta), where the opposite of theta is VE"
    d_weights = jnp.diag(D)
    
    # D-weighted inner product
    inner_product = jnp.sum(d_weights * V_true * V_vr)
    
    # D-weighted norms
    norm_true = jnp.sqrt(jnp.sum(d_weights * (V_true ** 2)))
    norm_vr = jnp.sqrt(jnp.sum(d_weights * (V_vr ** 2)))
    
    return inner_product / (norm_true * norm_vr + ε)

def get_lstd_weights(evaluator, network, params, random_policy, target_policy_fn = None):
    m = evaluator.num_actions
    def get_policy_matrix():
        if target_policy_fn is not None:
            pi_dist = target_policy_fn(evaluator.obs_stack)
        elif random_policy:
            pi_dist = distrax.Categorical(
                logits=jnp.zeros((evaluator.num_states, m))
            )
        else:
            pi_dist = network.apply(params, evaluator.obs_stack, method=network.policy)

        pi = pi_dist.probs
        terminal_policy = jnp.ones( [1,m], dtype=pi.dtype) / m
        pi = jnp.vstack([pi, terminal_policy])
        return pi
    
    # Get policy as S x A matrix
    pi = get_policy_matrix()
    # Get the value features:
    Φ = network.apply(params, evaluator.obs_stack, method=network.value_features)
    # terminal state
    Φ = jnp.vstack([Φ, jnp.zeros((1, Φ.shape[-1]))]) 
    
    # (add bias, but keep terminal state strictly zero):
    bias_col = jnp.ones((Φ.shape[0], 1)).at[-1].set(0.0)
    Φ = jnp.concatenate([Φ, bias_col], axis=-1)

    # Compute stationary dist (no terminal state)
    mu = evaluator.compute_stationary_distribution_raw(pi[:-1, :])
    mu = jnp.append(mu, 0.0)
    D = jnp.diag(mu) 

    # Get the exact formulation of the MDP
    γ = evaluator.gamma
    P = evaluator.P # 3d tensor S x A x S'
    P_π = jnp.einsum("sa,sam->sm", pi, P)
    R_π_s = jnp.einsum("sa,sa->s", pi, evaluator.R)
    # Gymnax awards the reward on the transition *INTO* s'
    R_π = P_π @ R_π_s
    V_lstd, w_lstd = LSTD_Exact(D, Φ, P_π, R_π, γ)
    return w_lstd    


def value_metrics(evaluator, network, params, random_policy=False, target_policy_fn = None):
    m = evaluator.num_actions
    def get_policy_matrix():
        if target_policy_fn is not None:
            pi_dist = target_policy_fn(evaluator.obs_stack)
        elif random_policy:
            pi_dist = distrax.Categorical(
                logits=jnp.zeros((evaluator.num_states, m))
            )
        else:
            pi_dist = network.apply(params, evaluator.obs_stack, method=network.policy)

        pi = pi_dist.probs
        terminal_policy = jnp.ones( [1,m], dtype=pi.dtype) / m
        pi = jnp.vstack([pi, terminal_policy])
        return pi
    
    # Get policy as S x A matrix
    pi = get_policy_matrix()
    # Get the value features:
    Φ = network.apply(params, evaluator.obs_stack, method=network.value_features)
    # terminal state
    Φ = jnp.vstack([Φ, jnp.zeros((1, Φ.shape[-1]))]) 
    
    # (add bias, but keep terminal state strictly zero):
    bias_col = jnp.ones((Φ.shape[0], 1)).at[-1].set(0.0)
    Φ = jnp.concatenate([Φ, bias_col], axis=-1)
    
    # Compute the true value:
    V_pi = evaluator.compute_true_values_raw(pi) 
    # Compute stationary dist (no terminal state)
    mu = evaluator.compute_stationary_distribution_raw(pi[:-1, :])
    mu = jnp.append(mu, 0.0)
    D = jnp.diag(mu) 

    # Get the exact formulation of the MDP
    γ = evaluator.gamma
    P = evaluator.P # 3d tensor S x A x S'
    P_π = jnp.einsum("sa,sam->sm", pi, P)
    R_π_s = jnp.einsum("sa,sa->s", pi, evaluator.R)
    # Gymnax awards the reward on the transition *INTO* s'
    R_π = P_π @ R_π_s
    I = jnp.eye(D.shape[-1])

    # Feature Quality (effective rank and PCA).
    _, S, _ = jnp.linalg.svd(Φ, full_matrices=False)
    sig_level = (1-γ) / 10.0
    effective_rank = jnp.sum(S > sig_level)
    # pca_phi = weighted_PCA(I, Φ)

    # Fits: Value Error, MSPBE
    V_nn = network.apply(params, evaluator.obs_stack, method=network.value)
    V_nn = jnp.append(V_nn, 0.0)
    
    V_lstd, w_lstd = LSTD_Exact(D, Φ, P_π, R_π, γ)
    V_vr, w_vr = LeastSquaresValue(D, Φ, V_pi)
    V_br, w_br = Bellman_Residual_Exact(D, Φ, P_π, R_π, γ)
    
    V_lstd_u, w_lstd_u = LSTD_Exact(I, Φ, P_π, R_π, γ)
    V_vr_u, w_vr_u = LeastSquaresValue(I, Φ, V_pi)
    V_br_u, w_br_u = Bellman_Residual_Exact(I, Φ, P_π, R_π, γ)

    # 1. Define configurations: (V, weight_mat, w)
    # Pass None for the weights of the neural network
    val_configs = {
        "LSTD": (V_lstd, D, w_lstd),
        "VR": (V_vr, D, w_vr),
        "BR": (V_br, D, w_br),
        "LSTD_uniform": (V_lstd_u, I, w_lstd_u),
        "VR_uniform": (V_vr_u, I, w_vr_u),
        "BR_uniform": (V_br_u, I, w_br_u),
        "nn": (V_nn, D, None) 
    }

    true_greedy_policy = compute_greedy_policy(P, R_π_s, γ, V_pi)

    # Get alignment:
    # Extract the correct keys from the dictionaries
    VE_VR = get_error_vectors(V_pi, val_configs["VR"][0], D, R_π, P_π, γ, Φ)['VE']
    nn_orthogonal_portion = get_error_vectors(V_pi, V_nn, D, R_π, P_π, γ, Φ)['Bellman_Orthogonal_Portion']

    # 1. Compute D-weighted dot product using element-wise multiplication with mu
    alignment_dot_product = jnp.sum(mu * VE_VR * nn_orthogonal_portion)
    negative_alignment = (alignment_dot_product < 0)
    
    # 2. Compute D-weighted norms for the denominator
    norm_VE_VR = jnp.sqrt(jnp.sum(mu * (VE_VR ** 2)))
    norm_nn_ortho = jnp.sqrt(jnp.sum(mu * (nn_orthogonal_portion ** 2)))
    
    # 3. Calculate true cosine similarity
    alignment = alignment_dot_product / (norm_VE_VR * norm_nn_ortho + 1e-8)

    # Consider the symmetry of the key matrix.
    # 1. Key Matrix A (State Space)
    A = D @ (jnp.eye(D.shape[0]) - γ * P_π)
    
    # 2. Symmetric and Skew-Symmetric components
    S = 0.5 * (A + A.T)
    K = 0.5 * (A - A.T)
    norm_s = jnp.linalg.norm(S, ord='fro')
    norm_k = jnp.linalg.norm(K, ord='fro')
    
    # 3. Precompute matrices for the alignment condition
    S_sq = S @ S
    SK_KS = (S @ K) - (K @ S)
    SA = S @ A 
    
    # 4. Check global positive definiteness of SA 
    # (If min eigenvalue > 0, SA is positive definite and E will globally decrease)
    # Using jnp.real to handle potential complex eigenvalues from numerical imprecision
    
    # 1. Extract the symmetric part of SA
    SA_symmetric = 0.5 * (SA + SA.T)
    
    # 2. Use 'eigvalsh' (the 'h' stands for Hermitian/Symmetric), which works perfectly on GPUs
    eigenvalues_SA = jnp.linalg.eigvalsh(SA_symmetric)
    
    # 3. Minimum eigenvalue
    min_eig_SA = jnp.min(eigenvalues_SA) # No jnp.real needed; symmetric eigenvalues are strictly real
    is_SA_pos_def = min_eig_SA > 0

    e = V_nn - V_pi
    term_1 = jnp.dot(e, S_sq @ e)
    term_2 = 0.5 * jnp.dot(e, SK_KS @ e)
    alignment_condition = term_1 + term_2 # If > 0, TD update decreases E
    alignment_condition_sign = alignment_condition > 0 # If > 0, TD update decreases E

    e_norm = e/jnp.linalg.norm(e)
    alignment_condition_normalized = jnp.dot(e_norm, S_sq @ e_norm) + 0.5 * jnp.dot(e_norm, SK_KS @ e_norm)

    # Compute the weighted value error E
    E = 0.5 * jnp.dot(e, A @ e)

    # 2. Initialize base metrics
    metrics = {
        "effective_rank": effective_rank,
        "capacity_angle": jnp.mean(get_capacity_angle(V_pi, val_configs["VR"][0], D)),
        "nn_lstd_diff": jnp.mean((val_configs["LSTD"][0] - V_nn)**2),
        "negative_alignment": negative_alignment,
        "alignment": alignment, # cosine similarity
        "value_grid": evaluator.get_value_grid(V_pi),
        "SA_min_eigenvalue": min_eig_SA,
        "is_SA_positive_definite": is_SA_pos_def,
        "alignment_condition": alignment_condition, # if zero, decreases E.
        "alignment_condition_normalized": alignment_condition_normalized, # if zero, decreases E.
        "E": E,
        "norm_s": norm_s,
        "norm_k": norm_k,
        "alignment_condition_sign": alignment_condition_sign
    }

    # 3. Iterate to compute Grids, Errors, Policies, MSEs, and Weights dynamically
    for prefix, (V, weight_mat, w) in val_configs.items():
        
        # Log weights if they exist
        if w is not None:
            metrics[f"{prefix}_weights"] = w

        # Generate grids for the primary methods
        if prefix in ["LSTD", "VR", "BR", "nn"]:
            evaluator.get_value_grid(V)

        # Compute error vectors
        errs = get_error_vectors(V_pi, V, weight_mat, R_π, P_π, γ, Φ)

        # Compute unweighted MSEs
        unweighted_mse = jax.tree.map(lambda x: jnp.mean(x**2), errs)
        for k, v in unweighted_mse.items():
            metrics[f"{prefix}_{k}"] = v

        # Compute weighted MSEs (Only for the primary D-weighted methods)
        if prefix in ["LSTD", "VR", "nn", "BR"]:
            weighted_mse = jax.tree.map(lambda x: jnp.sum(mu * x**2), errs)
            for k, v in weighted_mse.items():
                metrics[f"{prefix}_weighted_{k}"] = v

        # Compute Greedy Policy Accuracy
        greedy_pol = compute_greedy_policy(P, R_π_s, γ, V)
        metrics[f"{prefix}_greedy_correct"] = jnp.mean(true_greedy_policy == greedy_pol)

    return metrics