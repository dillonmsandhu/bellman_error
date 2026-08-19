def joint_td_E_loss(params):
    # --- TD Loss ---
    v_all = network.apply(params, S) 
    v_all = jnp.append(v_all, 0.0) 
    TD_targets_all = R_π + config['GAMMA'] * P_π @ v_all
    batch_targets = jax.lax.stop_gradient(TD_targets_all[batch_idx])
    
    v_batch = v_all[batch_idx]
    td_errors = v_batch - batch_targets
    td_loss = 0.5 * jnp.sum(batch_mu * weight_scale * (td_errors ** 2))

    # --- Exact Sampled E Loss ---
    e_all = V - v_all
    e_batch = e_all[batch_idx]
    
    # Get the transition probabilities for just the sampled states
    # Shape: (batch_size, n_states)
    P_batch = P_π[batch_idx, :] 
    
    # Compute the squared differences between the batch errors and ALL errors
    # Broadcasting: (batch_size, 1) - (1, n_states) -> (batch_size, n_states)
    e_diff_sq = (e_batch[:, None] - e_all[None, :]) ** 2
    
    # Compute the Laplacian term (weighted average over possible next states)
    # Shape: (batch_size,)
    laplacian_term = 0.5 * config['GAMMA'] * jnp.sum(P_batch * e_diff_sq, axis=1)
    
    # Compute the magnitude term
    # Shape: (batch_size,)
    magnitude_term = (1.0 - config['GAMMA']) * (e_batch ** 2)
    
    # Combine and sum over the batch
    E_loss = jnp.sum(batch_mu * weight_scale * (magnitude_term + laplacian_term))

    return td_loss + (config.get('BETA', 0.1) * E_loss)