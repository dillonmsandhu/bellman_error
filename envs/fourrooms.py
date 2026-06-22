import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any

class FourRoomsExactValue:
    """
    Exact policy evaluation for Four Rooms using GYMNAX COORDINATES [y, x].
    Replicates the Gymnax visual observation bug (transposed agent).
    """

    def __init__(
        self,
        size: int = 13,
        fail_prob: float = 0.01,
        gamma: float = 0.999,
        episodic: bool = True,
        use_visual_obs: bool = True,
        goal_pos: Tuple[int, int] | None = None,
        start_pos: Tuple[int, int] | None = None,
    ):
        self.N = int(size)
        self.fail_prob = float(fail_prob)
        self.gamma = float(gamma)
        self.episodic = episodic
        self.use_visual_obs = use_visual_obs

        # 1. Generate Map (H, W)
        self.env_map = self._generate_four_rooms_map()
        self.occupied_map = 1.0 - self.env_map.astype(jnp.float32)
        
        # 2. Coordinates: Gymnax uses Matrix Indexing (Row, Col) -> [y, x]
        y, x = jnp.where(self.env_map)
        self.coords = jnp.stack([y, x], axis=1).astype(jnp.int32) # Coordinates are (y,x)
        
        self.num_states = int(self.coords.shape[0])
        self.num_actions = 4
        self.terminal_idx = self.num_states
        self.num_total_states = self.num_states + 1
        
        # 3. Directions: Match Gymnax (y, x) definition
        # 0: Up (-1, 0), 1: Right (0, 1), 2: Down (1, 0), 3: Left (0, -1)
        self.directions = jnp.array(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32
        )

        # 4. Default Goal: Gymnax uses [8, 9] (Row 8, Col 9)
        default_goal = jnp.array([8, 9], dtype=jnp.int32)
        
        if goal_pos is None:
            is_valid = bool(self.env_map[default_goal[0], default_goal[1]])
            self.goal = default_goal if is_valid else self.coords[-1]
        else:
            # Assume user provides [y, x] if they are using this class
            g = jnp.array(goal_pos, dtype=jnp.int32)
            if not bool(self.env_map[g[0], g[1]]):
                raise ValueError(f"goal_pos={goal_pos} (y,x) is not a free cell.")
            self.goal = g
        
        if start_pos is None:
            # 5. Default Start: Gymnax uses [4, 1] (Row 4, Col 1)
            default_start = jnp.array([4, 1], dtype=jnp.int32)
        
            is_valid_start = bool(self.env_map[default_start[0], default_start[1]])
            self.start = default_start if is_valid_start else self.coords[0]
        else:
            self.start = jnp.array(start_pos, dtype=jnp.int32)
            # self.start = start_pos
            
        self.start_idx = self._coord_to_idx(self.start)
        self.goal_idx = self._coord_to_idx(self.goal)

        # 6. Build Observations (Replicating Gymnax Logic + Bug)
        self.obs_stack = self._build_obs_stack()
        
        # 7. Build Dynamics
        self.P, self.R = self._build_env_dynamics(continuing=False)
        self.P_cont, _ = self._build_env_dynamics(continuing=True)

    def _generate_four_rooms_map(self) -> jax.Array:
        four_rooms_str = """
xxxxxxxxxxxxx
x     x     x
x     x     x
x           x
x     x     x
x     x     x
xx xxxx     x
x     xxx xxx
x     x     x
x     x     x
x           x
x     x     x
xxxxxxxxxxxxx"""
        lines = four_rooms_str.strip().split("\n")
        bool_map = []
        for row in lines:
            bool_map.append([r == " " for r in row])
        return jnp.array(bool_map)

    def _coord_to_idx(self, coord: jax.Array) -> int:
        match = jnp.all(self.coords == coord[None, :], axis=1)
        return int(jnp.argmax(match))
    
    def _build_obs_stack(self) -> jax.Array:
        """
        Constructs observations. 
        CRITICAL: Replicates Gymnax 'Mirror World' Bug for Visual Obs.
        """
        if self.use_visual_obs:
            # --- Visual Observation ---
            # Output Shape: (N_states, Height, Width, 2)
            
            # 1. Initialize Agent Maps
            agent_maps = np.zeros((self.num_states, self.N, self.N), dtype=np.float32)
            
            # 2. Set Agent Positions
            # self.coords is [y, x] (Row, Col).
            y_coords = self.coords[:, 0]
            x_coords = self.coords[:, 1]
            
            # GYMNAX LOGIC: agent_map.at[pos[1], pos[0]].set(1)
            # pos[1] is x (Col), pos[0] is y (Row).
            # So it writes to grid[x, y].
            # We replicate this transposition:
            agent_maps[np.arange(self.num_states), x_coords, y_coords] = 1.0

            # 3. Wall Map (Gymnax uses standard map for walls)
            wall_stack = np.broadcast_to(self.occupied_map, agent_maps.shape)

            # 4. Stack
            obs = np.stack([wall_stack, agent_maps], axis=-1)
            return jnp.asarray(obs, dtype=jnp.float32)

        else:
            # --- Vector Observation ---
            # Gymnax returns: [pos[0], pos[1], goal[0], goal[1]]
            # Which is [y, x, y, x]
            
            pos_yx = self.coords
            
            goal_yx = self.goal
            goal_stack = np.broadcast_to(goal_yx[None, :], pos_yx.shape)
            
            return jnp.asarray(np.concatenate([pos_yx, goal_stack], axis=1), dtype=jnp.float32)

    def _step_pos(self, pos: jax.Array, action: int) -> jax.Array:
        # pos is [y, x], directions are [dy, dx]
        proposed = pos + self.directions[action]
        
        # Check collision in map[y, x]
        # proposed[0] is y, proposed[1] is x
        can_move = self.env_map[proposed[0], proposed[1]]
        
        return jax.lax.select(can_move, proposed, pos)

    def _build_env_dynamics(self, continuing: bool) -> Tuple[jax.Array, jax.Array]:
        """
        Builds P and R.
        
        LOGIC:
        1. Rewards: Shifted to 'Exit' to ensure V(Goal) = 1.0.
           - V(Goal) = R_exit (1.0) + gamma*0 = 1.0
           - V(Neighbor) = R_entry (0.0) + gamma*V(Goal) = 0.99
           
        2. Dynamics: MATCHES GYMNAX EXACTLY.
           - P(intended) = 1 - fail_prob
           - P(wrong)    = fail_prob / 3
        """
        P = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)
        R = np.zeros((self.num_total_states, self.num_actions), dtype=np.float32)

        p_correct = 1.0 - self.fail_prob
        p_wrong = self.fail_prob / 3.0

        for s_idx in range(self.num_states):
            pos = self.coords[s_idx] # (y, x)

            # --- 1. GOAL STATE (Source of Value 1.0) ---
            if s_idx == self.goal_idx:
                if not continuing:
                    # Episodic: Goal -> Terminal
                    P[s_idx, :, self.terminal_idx] = 1.0
                    R[s_idx, :] = 1.0  # Reward 1.0 on Exit
                else:
                    # Continuing: Goal -> Start
                    P[s_idx, :, self.start_idx] = 1.0
                    R[s_idx, :] = 1.0
                continue

            # --- 2. STANDARD STATES ---
            for chosen_a in range(self.num_actions):
                for executed_a in range(self.num_actions):
                    # Use the Gymnax Probability Logic
                    prob = p_correct if executed_a == chosen_a else p_wrong
                    
                    if prob == 0: continue
                        
                    next_pos = self._step_pos(pos, executed_a)
                    hits_goal = bool(jnp.all(next_pos == self.goal))
                    
                    if hits_goal:
                        next_idx = self.goal_idx
                        P[s_idx, chosen_a, next_idx] += prob
                        # Reward 0.0 on Entry (Value comes from next state)
                        R[s_idx, chosen_a] += prob * 0.0
                    else:
                        next_idx = self._coord_to_idx(next_pos)
                        P[s_idx, chosen_a, next_idx] += prob
                        # Standard step reward is 0

        # Terminal state dynamics
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        
        return jnp.asarray(P), jnp.asarray(R)

    # def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array) -> jax.Array:
    #     P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
    #     R_pi = jnp.einsum("sa,sa->s", pi, R_env)
    #     A = jnp.eye(self.num_total_states) - self.gamma * P_pi
    #     return jnp.linalg.solve(A, R_pi)

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
