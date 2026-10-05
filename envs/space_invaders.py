from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Any, Dict, List


class SpaceInvadersExactValue:
    """
    Exact policy evaluation for a 2-Alien Tabular Space Invaders Environment.
    
    State representation:
        - 2 Aliens: Alien 0 and Alien 1 in a 1x2 formation.
        - Grid dimensions: Width W = 5 columns, Height H = 4 rows.
        - Rows:
            - y = 0: Top row (spawn row).
            - y = 1: Mid row.
            - y = 2: Low danger row (aliens directly above player drop point-blank bombs).
            - y = 3: Defense line (player stationed here; aliens reaching here trigger INVASION/DEATH).
        - Player: column x_p in {0, 1, 2, 3, 4}.
        - 4 Actions: 0: No-op, 1: Left, 2: Right, 3: Fire.
        - Termination:
            - WIN: Both aliens destroyed (+1.0 cumulative reward).
            - DEATH: Aliens invade defense line (y=3) or bomb player at point-blank range (0.0 reward).
            
    State count:
        - Both alive (1, 1): 5 (player) * 3 (rows) * 4 (cols) * 2 (dirs) = 120 states.
        - Only A0 alive (1, 0): 5 * 3 * 5 * 2 = 150 states.
        - Only A1 alive (0, 1): 5 * 3 * 5 * 2 = 150 states.
        - Total active states: 120 + 150 + 150 = 420 states.
        - Terminal state: 1 absorbing sink state (index 420).
        - Total states in P tensor: 421 states.
    """

    def __init__(
        self,
        width: int = 5,
        height: int = 6,
        gamma: float = 0.99,
        episodic: bool = True,
        use_visual_obs: bool = True,
        endless: bool = True,
        kill_reward: float = 0.5,
        death_penalty: float = 0.0,
    ):
        self.W = int(width)
        self.H = int(height)
        self.gamma = float(gamma)
        self.episodic = episodic
        self.use_visual_obs = use_visual_obs
        self.endless = endless
        self.kill_reward = float(kill_reward)
        self.death_penalty = float(death_penalty)
        self.fail_prob = 0.0

        self.num_actions = 4
        self.action_names = ["No-op", "Left", "Right", "Fire"]
        self.directions = jnp.array(
            [[0, 0], [0, -1], [0, 1], [-1, 0]], dtype=jnp.float32
        )

        # 1. Enumerate all valid active state configurations
        self.states: List[Tuple[int, int, int, int, int, int, int]] = []
        
        # Config 1: Both alive (1, 1)
        # col0 in 0..W-2, col1 = col0 + 1
        for xp in range(self.W):
            for fy in range(self.H - 1): # 0, 1, 2
                for col0 in range(self.W - 1): # 0, 1, 2, 3
                    for fdir in [-1, 1]:
                        self.states.append((1, 1, xp, fy, col0, col0 + 1, fdir))

        # Config 2: Only A0 alive (1, 0)
        for xp in range(self.W):
            for fy in range(self.H - 1):
                for col0 in range(self.W):
                    for fdir in [-1, 1]:
                        self.states.append((1, 0, xp, fy, col0, -1, fdir))

        # Config 3: Only A1 alive (0, 1)
        for xp in range(self.W):
            for fy in range(self.H - 1):
                for col1 in range(self.W):
                    for fdir in [-1, 1]:
                        self.states.append((0, 1, xp, fy, -1, col1, fdir))

        self.num_states = len(self.states)
        self.terminal_idx = self.num_states
        self.num_total_states = self.num_states + 1

        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        self.coords = jnp.array(self.states, dtype=jnp.int32)

        # Canonical initial state: both alive, player center, aliens top row, marching right
        # (1, 1, 2, 0, 1, 2, 1)
        start_state = (1, 1, self.W // 2, 0, 1, 2, 1)
        self.start_idx = self.state_to_idx[start_state]
        self.goal_idx = self.start_idx
        self.reset_indices = jnp.array([self.start_idx], dtype=jnp.int32)
        self.occupied_map = jnp.zeros((self.H, self.W), dtype=jnp.float32)

        # 2. Build Observations
        self.obs_stack = self._build_obs_stack()

        # 3. Build Dynamics P and R (Episodic and Continuing)
        P_ep, R_ep, P_win, P_death = self._build_env_dynamics(continuing=False)
        P_cnt, R_cnt, _, _ = self._build_env_dynamics(continuing=True)

        self.P = jnp.asarray(P_ep, dtype=jnp.float32)
        self.R = jnp.asarray(R_ep, dtype=jnp.float32)
        self.P_cont = jnp.asarray(P_cnt, dtype=jnp.float32)
        self.P_win = jnp.asarray(P_win, dtype=jnp.float32)
        self.P_death = jnp.asarray(P_death, dtype=jnp.float32)

    def _build_obs_stack(self) -> jax.Array:
        """
        Builds spatial grid observations of shape (num_states, H, W, 2).
        Channel 0: Player position map (1.0 at [3, x_p]).
        Channel 1: Alien position map (1.0 at living alien coordinates [fy, col]).
        """
        if self.use_visual_obs:
            obs = np.zeros((self.num_states, self.H, self.W, 2), dtype=np.float32)
            for i, (a0, a1, xp, fy, col0, col1, fdir) in enumerate(self.states):
                # Channel 0: Player at row H - 1
                obs[i, self.H - 1, xp, 0] = 1.0
                # Channel 1: Living aliens
                if a0 == 1:
                    obs[i, fy, col0, 1] = 1.0
                if a1 == 1:
                    obs[i, fy, col1, 1] = 1.0
            return jnp.asarray(obs, dtype=jnp.float32)
        else:
            # Normalized feature vector: [xp/W, fy/H, col0/W, col1/W, fdir, a0, a1]
            obs = np.zeros((self.num_states, 7), dtype=np.float32)
            for i, (a0, a1, xp, fy, col0, col1, fdir) in enumerate(self.states):
                obs[i] = [
                    xp / float(self.W),
                    fy / float(self.H),
                    max(0, col0) / float(self.W),
                    max(0, col1) / float(self.W),
                    1.0 if fdir == 1 else 0.0,
                    float(a0),
                    float(a1),
                ]
            return jnp.asarray(obs, dtype=jnp.float32)

    def _build_env_dynamics(
        self, continuing: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs the exact transition tensor P and reward tensor R.
        """
        P = np.zeros(
            (self.num_total_states, self.num_actions, self.num_total_states),
            dtype=np.float32,
        )
        R = np.zeros(
            (self.num_total_states, self.num_actions, self.num_total_states),
            dtype=np.float32,
        )
        P_win = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        P_death = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        for s_idx in range(self.num_states):
            a0, a1, xp, fy, col0, col1, fdir = self.idx_to_state[s_idx]

            for a in range(self.num_actions):
                # 1. Player move
                if a == 0:
                    xp_next = xp
                elif a == 1:
                    xp_next = max(0, xp - 1)
                elif a == 2:
                    xp_next = min(self.W - 1, xp + 1)
                elif a == 3:  # Fire
                    xp_next = xp

                # 2. Shooting resolution
                a0_next = a0
                a1_next = a1
                reward = 0.0

                if a == 3:  # Laser shot straight up column xp
                    hit_a0 = (a0 == 1 and xp == col0)
                    hit_a1 = (a1 == 1 and xp == col1)

                    if hit_a0:
                        a0_next = 0
                        reward += self.kill_reward
                    elif hit_a1:
                        a1_next = 0
                        reward += self.kill_reward

                # Check WIN: both aliens dead
                if a0_next == 0 and a1_next == 0:
                    P_win[s_idx, a] = 1.0
                    if not self.endless:
                        if not continuing:
                            P[s_idx, a, self.terminal_idx] = 1.0
                            R[s_idx, a, self.terminal_idx] = reward
                        else:
                            P[s_idx, a, self.start_idx] = 1.0
                            R[s_idx, a, self.start_idx] = reward
                    else:
                        # Endless wave mode: new wave respawns at top row, player retains horizontal position
                        s_respawn = (1, 1, xp_next, 0, 1, 2, 1)
                        respawn_idx = self.state_to_idx[s_respawn]
                        P[s_idx, a, respawn_idx] = 1.0
                        R[s_idx, a, respawn_idx] = reward
                    continue

                # 3. Fleet movement of surviving alien(s)
                if a0_next == 1 and a1_next == 1:
                    # 2-alien squad moving together
                    if fdir == 1:
                        if col1 < self.W - 1:
                            col0_next = col0 + 1
                            col1_next = col1 + 1
                            fy_next = fy
                            fdir_next = 1
                        else:  # Hit right wall -> drop down and reverse
                            col0_next = col0
                            col1_next = col1
                            fy_next = fy + 1
                            fdir_next = -1
                    else:  # fdir == -1
                        if col0 > 0:
                            col0_next = col0 - 1
                            col1_next = col1 - 1
                            fy_next = fy
                            fdir_next = -1
                        else:  # Hit left wall -> drop down and reverse
                            col0_next = col0
                            col1_next = col1
                            fy_next = fy + 1
                            fdir_next = 1

                elif a0_next == 1 and a1_next == 0:
                    # Only A0 alive
                    col1_next = -1
                    if fdir == 1:
                        if col0 < self.W - 1:
                            col0_next = col0 + 1
                            fy_next = fy
                            fdir_next = 1
                        else:
                            col0_next = col0
                            fy_next = fy + 1
                            fdir_next = -1
                    else:
                        if col0 > 0:
                            col0_next = col0 - 1
                            fy_next = fy
                            fdir_next = -1
                        else:
                            col0_next = col0
                            fy_next = fy + 1
                            fdir_next = 1

                else:  # a0_next == 0 and a1_next == 1
                    # Only A1 alive
                    col0_next = -1
                    if fdir == 1:
                        if col1 < self.W - 1:
                            col1_next = col1 + 1
                            fy_next = fy
                            fdir_next = 1
                        else:
                            col1_next = col1
                            fy_next = fy + 1
                            fdir_next = -1
                    else:
                        if col1 > 0:
                            col1_next = col1 - 1
                            fy_next = fy
                            fdir_next = -1
                        else:
                            col1_next = col1
                            fy_next = fy + 1
                            fdir_next = 1

                # 4. Check DEATH / INVASION
                # Invasion: aliens reached player row (fy_next >= 3)
                is_invaded = fy_next >= self.H - 1

                # Point-blank bombing: living alien at row H-2 directly above player's new position
                is_bombed = (fy_next == self.H - 2) and (
                    (a0_next == 1 and col0_next == xp_next)
                    or (a1_next == 1 and col1_next == xp_next)
                )

                if is_invaded or is_bombed:
                    P_death[s_idx, a] = 1.0
                    death_rew = reward - self.death_penalty
                    if not continuing:
                        P[s_idx, a, self.terminal_idx] = 1.0
                        R[s_idx, a, self.terminal_idx] = death_rew
                    else:
                        P[s_idx, a, self.start_idx] = 1.0
                        R[s_idx, a, self.start_idx] = death_rew
                    continue

                # 5. Normal in-game transition
                s_next = (
                    a0_next,
                    a1_next,
                    xp_next,
                    fy_next,
                    col0_next,
                    col1_next,
                    fdir_next,
                )
                s_next_idx = self.state_to_idx[s_next]
                P[s_idx, a, s_next_idx] = 1.0
                R[s_idx, a, s_next_idx] = reward

        # Terminal state absorbing
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        R[self.terminal_idx, :, self.terminal_idx] = 0.0

        return P, R, P_win, P_death

    def solve_linear_system(
        self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array
    ) -> jax.Array:
        """Solves the exact linear system (I - gamma * P_pi) V = R_pi."""
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        R_pi = jnp.einsum("sa,sam,sam->s", pi, P_env, R_env)
        A = jnp.eye(self.num_total_states) - self.gamma * P_pi
        return jnp.linalg.solve(A, R_pi)

    def compute_true_values_raw(self, pi: jax.Array) -> jax.Array:
        """Returns vector of exact state values V(s) of shape (num_total_states,)."""
        return self.solve_linear_system(pi, self.P, self.R)

    def compute_true_values(self, pi: jax.Array) -> jax.Array:
        return self.compute_true_values_raw(pi)

    def compute_stationary_distribution_raw(
        self, pi: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Computes the stationary distribution mu under the resetting chain P_cont.
        Returns (mu, P_pi).
        """
        P_env = self.P_cont[: self.num_states, :, : self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)

        A = P_pi.T - jnp.eye(self.num_states)
        A = A.at[-1, :].set(1.0)
        b = jnp.zeros(self.num_states).at[-1].set(1.0)

        mu = jnp.linalg.solve(A, b)
        mu = jnp.clip(mu, a_min=0.0)
        return mu / mu.sum(), P_pi

    def compute_discounted_visitation_raw(self, pi: jax.Array) -> jax.Array:
        """
        Computes the normalized discounted occupancy distribution:
            d_gamma = (1 - gamma) * mu_0 * (I - gamma * P_pi_trans)^{-1}
        """
        P_env = self.P[: self.num_states, :, : self.num_states]
        P_pi = jnp.einsum("sa,sam->sm", pi, P_env)
        mu_0 = jnp.zeros(self.num_states).at[self.start_idx].set(1.0)

        A = jnp.eye(self.num_states) - self.gamma * P_pi.T
        d_gamma = jnp.linalg.solve(A, (1.0 - self.gamma) * mu_0)
        return d_gamma / jnp.sum(d_gamma)

    def compute_absorption_probabilities(
        self, pi: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Computes exact closed-form probability of WIN vs DEATH from each state:
            prob_win(s) = [(I - P_pi_trans)^{-1} P_win_pi]_s
            prob_death(s) = [(I - P_pi_trans)^{-1} P_death_pi]_s
        """
        P_trans = self.P[: self.num_states, :, : self.num_states]
        P_pi_trans = jnp.einsum("sa,sam->sm", pi[: self.num_states], P_trans)
        I = jnp.eye(self.num_states)
        fundamental = jnp.linalg.inv(I - P_pi_trans)

        P_win_pi = jnp.einsum("sa,sa->s", pi[: self.num_states], self.P_win)
        P_death_pi = jnp.einsum("sa,sa->s", pi[: self.num_states], self.P_death)

        prob_win = fundamental @ P_win_pi
        prob_death = fundamental @ P_death_pi
        return prob_win, prob_death

    def get_optimal_value_function(
        self, tol: float = 1e-6, max_iters: int = 1000
    ) -> jax.Array:
        """Computes V* via exact Bellman optimality value iteration."""
        V = jnp.zeros(self.num_total_states)
        R_expected = jnp.einsum("sam,sam->sa", self.P, self.R)

        def body_fun(val):
            i, V_curr, delta = val
            expected_v = jnp.einsum("sam,m->sa", self.P, V_curr)
            Q = R_expected + self.gamma * expected_v
            V_new = jnp.max(Q, axis=-1)
            V_new = V_new.at[self.terminal_idx].set(0.0)
            delta = jnp.max(jnp.abs(V_new - V_curr))
            return (i + 1, V_new, delta)

        def cond_fun(val):
            i, V_curr, delta = val
            return jnp.logical_and(i < max_iters, delta > tol)

        _, V_star, _ = jax.lax.while_loop(cond_fun, body_fun, (0, V, 1.0))
        return V_star

    def get_value_grid(self, values: jax.Array) -> jax.Array:
        """
        Projects per-state values onto a 2D (H, W) grid representing average value
        conditioned on the player's horizontal position.
        """
        if values.shape[0] == self.num_total_states:
            values = values[: self.num_states]

        # Aggregate values by player position (xp) and lowest alien row (fy)
        grid = jnp.zeros((self.H, self.W), dtype=values.dtype)
        counts = jnp.zeros((self.H, self.W), dtype=values.dtype)

        xp_coords = self.coords[:, 2]
        fy_coords = self.coords[:, 3]

        for i in range(self.num_states):
            grid = grid.at[fy_coords[i], xp_coords[i]].add(values[i])
            counts = counts.at[fy_coords[i], xp_coords[i]].add(1.0)

        return jnp.where(counts > 0, grid / counts, 0.0)
