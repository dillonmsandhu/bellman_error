import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any

class WhirlpoolExactValue:
    """
    Exact policy evaluation for the Whirlpool Environment using GYMNAX COORDINATES [y, x].
    A highly rotationally-asymmetric MDP designed to stress-test TD learning's K matrix.
    """

    def __init__(
        self,
        size: int = 13,
        gamma: float = 0.999,
        episodic: bool = True,
        use_visual_obs: bool = True,
        fail_prob: float = 0.9
    ):
        self.N = int(size)
        assert self.N % 2 == 1, "must be of size odd x odd, gave an even number."
        self.gamma = float(gamma)
        self.episodic = episodic
        self.use_visual_obs = use_visual_obs
        self.fail_prob = fail_prob

        # 1. Generate Map (H, W) - Fully open grid for the whirlpool
        self.env_map = jnp.ones((self.N, self.N), dtype=bool)
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)
        
        # 2. Coordinates: Gymnax uses Matrix Indexing (Row, Col) -> [y, x]
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32) 
        
        self.num_states = int(self.coords.shape[0])
        self.num_actions = 4
        self.terminal_idx = self.num_states
        self.num_total_states = self.num_states + 1
        
        # 3. Directions
        # 0: Up (-1, 0), 1: Right (0, 1), 2: Down (1, 0), 3: Left (0, -1)
        self.directions = jnp.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32
        )
        # 4. Goal is strictly the center
        center_idx = self.N // 2
        self.goal = jnp.array([center_idx, center_idx], dtype=jnp.int32)
        self.goal_idx = self._coord_to_idx(self.goal)
        
        # 5. Determine the out ring (for uniform random reset)
        outer_ring_coords = []
        outer_ring_indices = []
        max_d = self.N // 2
        
        for i in range(self.num_states):
            py, px = int(self.coords[i][0]), int(self.coords[i][1])
            if max(abs(py - center_idx), abs(px - center_idx)) == max_d:
                outer_ring_coords.append([py, px])
                outer_ring_indices.append(i)
                
        self.outer_ring_coords = jnp.array(outer_ring_coords, dtype=jnp.int32)
        self.outer_ring_indices = jnp.array(outer_ring_indices, dtype=jnp.int32)
        

        self.obs_stack = self._build_obs_stack()
        # 7. Build Dynamics
        self.P, self.R = self._build_env_dynamics(continuing=False)
        self.P_cont, _ = self._build_env_dynamics(continuing=True)

    def _coord_to_idx(self, coord: jax.Array) -> int:
        match = jnp.all(self.coords == coord[None, :], axis=1)
        return int(jnp.argmax(match))
    
    def _build_obs_stack(self) -> jax.Array:
        if self.use_visual_obs:
            agent_maps = np.zeros((self.num_states, self.N, self.N), dtype=np.float32)
            y_coords = self.coords[:, 0]
            x_coords = self.coords[:, 1]
            agent_maps[np.arange(self.num_states), x_coords, y_coords] = 1.0
            wall_stack = np.broadcast_to(self.occupied_map, agent_maps.shape)
            obs = np.stack([wall_stack, agent_maps], axis=-1)
            return jnp.asarray(obs, dtype=jnp.float32)
        else:
            pos_yx = self.coords
            goal_stack = np.broadcast_to(self.goal[None, :], pos_yx.shape)
            return jnp.asarray(np.concatenate([pos_yx, goal_stack], axis=1), dtype=jnp.float32)

    def _step_pos(self, pos: jax.Array, action: int) -> jax.Array:
        proposed = pos + self.directions[action]
        # Keep within bounds just in case, though logic should prevent out-of-bounds
        proposed = jnp.clip(proposed, 0, self.N - 1)
        return proposed

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        P = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)
        R = np.zeros((self.num_total_states, self.num_actions), dtype=np.float32)

        cy, cx = self.N // 2, self.N // 2
        max_d = self.N // 2

        for s_idx in range(self.num_states):
            pos = self.coords[s_idx]
            
            # Cast to native int for safe Python logic and math
            y, x = int(pos[0]), int(pos[1])

            # --- 1. GOAL STATE ---
            if s_idx == self.goal_idx:
                if not continuing:
                    P[s_idx, :, self.terminal_idx] = 1.0
                    R[s_idx, :] = 1.0  
                else:
                    # Reset uniformly to the outermost ring to guarantee symmetric mu!
                    for idx in self.outer_ring_indices:
                        P[s_idx, :, idx] = 1/len(self.outer_ring_indices)
                    R[s_idx, :] = 1.0
                continue

            # --- 2. DETERMINE RING DISTANCE ---
            d = max(abs(y - cy), abs(x - cx))

            # --- 3. MAP ACTIONS (With exact corner handling) ---
            # 0: Up, 1: Right, 2: Down, 3: Left
            if y == cy - d and x < cx + d:     # Top edge
                cw, ccw, inward, outward = 1, 3, 2, 0
            elif x == cx + d and y < cy + d:   # Right edge
                cw, ccw, inward, outward = 2, 0, 3, 1
            elif y == cy + d and x > cx - d:   # Bottom edge
                cw, ccw, inward, outward = 3, 1, 0, 2
            elif x == cx - d and y > cy - d:   # Left edge
                cw, ccw, inward, outward = 0, 2, 1, 3
            else: # Fallback (Bottom-Left Corner edge case)
                cw, ccw, inward, outward = 0, 2, 1, 3

            # 4. Resolve Actions based on Physics Rules
            for action in range(self.num_actions):
                transitions = [] # (Probability, Executed Cardinal Action)
                
                if action == cw:
                    transitions.append((1.0, cw))
                elif action == ccw:
                    transitions.append((1-self.fail_prob, ccw))
                    transitions.append((self.fail_prob, cw))
                elif action == outward:
                    transitions.append((1.0, cw))
                elif action == inward:
                    transitions.append((1-self.fail_prob, inward))
                    transitions.append((self.fail_prob, cw))

                # Apply Transitions
                for prob, executed_a in transitions:
                    next_pos = self._step_pos(pos, executed_a)
                    hits_goal = bool(jnp.all(next_pos == self.goal))
                    
                    if hits_goal:
                        P[s_idx, action, self.goal_idx] += prob
                    else:
                        next_idx = self._coord_to_idx(next_pos)
                        P[s_idx, action, next_idx] += prob

        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        return jnp.asarray(P), jnp.asarray(R)

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
        # 1. Get the state-to-state transition matrix under the policy
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        
        # 2. Get the original state-dependent reward (1.0 only at goal state)
        R_pi_delayed = jnp.einsum("sa,sa->s", pi, R_env)
        
        # 3. Shift the reward to the entry transition!
        R_pi_shifted = P_pi @ R_pi_delayed
        
        # 4. Solve the standard linear system using the shifted rewards
        A = jnp.eye(self.num_total_states) - self.gamma * P_pi
        return jnp.linalg.solve(A, R_pi_shifted)

    # def get_value_grid(self, values: jax.Array) -> jax.Array:
    #         """
    #         Map per-state values to N x N grid.
    #         MATCHES VISUAL OBS: Writes to grid[x, y] (Transposed).
    #         """
    #         if values.shape[0] == self.num_total_states:
    #             values = values[: self.num_states]

    #         grid = jnp.zeros((self.N, self.N), dtype=values.dtype)
            
    #         # self.coords is [y, x] (Row, Col)
    #         # Visual Obs puts agent at [x, y].
    #         # match visual obs for comparison.
    #         return grid.at[self.coords[:, 1], self.coords[:, 0]].set(values)


    def get_value_grid(self, values: jax.Array) -> jax.Array:
        """
        Map per-state values to N x N grid.
        Uses standard matrix indexing [row, col] (which is [y, x]).
        """
        if values.shape[0] == self.num_total_states:
            values = values[: self.num_states]

        grid = jnp.zeros((self.N, self.N), dtype=values.dtype)
        
        # self.coords is already [y, x] (Row, Col)
        # We now place them correctly into grid[row, col]
        return grid.at[self.coords[:, 0], self.coords[:, 1]].set(values)

    def compute_true_values_raw(self, pi: jax.Array) -> Tuple[jax.Array, jax.Array, Any]:
        "returns vector"
        V_pi = self.solve_linear_system(pi, self.P, self.R)
        return V_pi
    
    def compute_true_values(self, pi: jax.Array) -> Tuple[jax.Array, jax.Array, Any]:
        "Returns grid"
        V_pi = self.compute_true_values_raw(pi)
        return self.get_value_grid(V_pi)

    def compute_stationary_distribution_raw(self, pi: jax.Array) -> jax.Array:
        "Returns vector"
        P_env = self.P_cont[:self.num_states, :, :self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        A = P_pi.T - jnp.eye(self.num_states)
        A = A.at[-1, :].set(1.0)
        b = jnp.zeros(self.num_states)
        b = b.at[-1].set(1.0)
        mu = jnp.linalg.solve(A, b)
        mu = jnp.clip(mu, a_min=0.0)
        return mu / mu.sum()

    def compute_stationary_distribution(self, pi: jax.Array) -> jax.Array:
        "Returns grid"
        mu = self.compute_stationary_distribution_raw(pi)
        return self.get_value_grid(mu)
    
    def compute_v_error_on_d_pi(self, pi: jax.Array, v_pred: jax.Array, v_true: jax.Array):
        mu = self.compute_stationary_distribution(pi)
        valid_mask = self.occupied_map == 0
        mu_flat = mu[valid_mask]
        v_pred_flat = v_pred[valid_mask]
        v_true_flat = v_true[valid_mask]
        return jnp.sum(mu_flat * (v_pred_flat - v_true_flat)**2)
        
    def compute_discounted_visitation_raw(self, pi: jax.Array) -> jax.Array:
        P_env = self.P[:self.num_states, :, :self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        rho_0 = jnp.zeros(self.num_states)
        rho_0 = rho_0.at[self.start_idx].set(1.0)
        A = jnp.eye(self.num_states) - self.gamma * P_pi.T
        d_gamma = jnp.linalg.solve(A, (1 - self.gamma) * rho_0)
        d_gamma_norm = d_gamma / jnp.sum(d_gamma)
        return d_gamma_norm
    
    def compute_discounted_visitation(self, pi: jax.Array) -> jax.Array:
        mu = self.compute_discounted_visitation_raw(pi)
        return self.get_value_grid(mu)
