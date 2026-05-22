from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.bellman_error as bellman_error
from core.networks import nn
# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "ppo_iv_z"

class IVActorCritic(nn.Module):
    action_dim: int
    is_continuous: bool
    norm_type: str = 'None'
    z_dim: int = 32
    phi_dim: int = 32

    def setup(self):
        self.actor_torso = networks.PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.phi_dim)
        self.phi_net = networks.PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.phi_dim)
        self.z_net = networks.PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.phi_dim)
        
        self.pi_head = networks.PolicyHead(action_dim=self.action_dim, is_continuous=self.is_continuous)
        self.w_layer = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init = nn.initializers.zeros)
        
        # g_A will automatically size its input kernel to (z_dim + action_dim) 
        # on the first forward pass.
        self.z_to_phi = nn.Dense(self.phi_dim)
        self.g_A_dense1 = nn.Dense(self.phi_dim)
        self.g_A_norm = nn.LayerNorm(use_scale=False, use_bias=False)
        self.g_A_dense2 = nn.Dense(self.phi_dim) # predicts td-diff
        self.g_A_dense_done = nn.Dense(1) # predicts done

    def w(self, phi_features):
        return self.w_layer(phi_features)
    
    def value(self, obs):
        phi_curr = self.phi(obs)
        value = self.w(phi_curr).squeeze(-1)
        return value

    def policy(self, x):
        return self.pi_head(self.actor_torso(x))
    
    def value_features(self, obs):
        return self.phi(obs)
    
    def phi(self, obs):
        return self.phi_net(obs)

    def z(self, obs):
        return self.z_net(obs)
    
    def g_A(self, z_and_a):
        # Extract the pure instrument z from the concatenated [z, a]
        z = z_and_a[..., :self.z_dim] 
        
        # 1. Predict the current features from the instrument
        phi_hat = self.z_to_phi(z)
        
        # 2. Predict the discounted expected future features
        x = self.g_A_dense1(z_and_a)
        x = jax.nn.leaky_relu(x)
        # x = self.g_A_norm(x) # (Optional, but usually helpful for dynamics)
        gamma_phi_prime = self.g_A_dense2(x)
        
        # 3. The mathematically pure Residual Connection
        x_hat = phi_hat - gamma_phi_prime 
        
        # Done prediction branches off the dynamics hidden state
        done_logit = self.g_A_dense_done(x)
        
        return x_hat, done_logit

    # --- Action optional ---
    def __call__(self, obs, action=None):
        pi = self.policy(obs)
        phi_curr = self.phi(obs)
        value = self.w(phi_curr).squeeze(-1)
        
        # for initialization only...
        if action is not None:
            _z = self.z(obs)
            if _z.ndim > action.ndim:
                _z = _z[0]
            _z_and_a = jnp.concatenate([_z, action], axis=-1)
            _g_a_out = self.g_A(_z_and_a)
            
        return pi, value

def one_hot_action(traj_batch, is_continuous, action_dim):
    action_features = traj_batch.action
    
    if not is_continuous: # e.g., CartPole
        action_features = jax.nn.one_hot(traj_batch.action, num_classes=action_dim)
    
    return action_features

def vic_loss_fn(phi_batch):
    # Center the batch
    phi_centered = phi_batch - jnp.mean(phi_batch, axis=0)
    
    # 1. Variance Loss: push standard deviation of each feature towards 1
    std = jnp.sqrt(jnp.var(phi_batch, axis=0) + 1e-4)
    var_loss = jnp.mean(jnp.maximum(0.0, 1.0 - std))
    
    # 2. Covariance Loss: penalize off-diagonal correlations
    cov_matrix = (phi_centered.T @ phi_centered) / (phi_batch.shape[0] - 1)
    off_diagonal = cov_matrix - jnp.diag(jnp.diag(cov_matrix))
    cov_loss = jnp.sum(off_diagonal ** 2) / phi_batch.shape[-1]
    
    return var_loss, cov_loss

def iv_loss_fn(params, network, traj_batch, advantages, targets, config):
    # 1. Forward Passes
    z = network.apply(params, traj_batch.obs, method=network.z)
    # z = network.apply(params, traj_batch.obs, method=network.phi)
    action_features = one_hot_action(traj_batch, config['IS_CONTINUOUS'], network.action_dim)
    z = jnp.concatenate([z, action_features], axis=-1)
    
    phi_curr = network.apply(params, traj_batch.obs, method=network.phi)

    # =========================================================
    # STAGE 1: Dynamics Distillation (Updates g_A and phi_net)
    # =========================================================
    # z is detached so Stage 1 doesn't corrupt the instrument with information about phi_next.
    x_hat, done_logit = network.apply(params, z, method=network.g_A)
    
    # task 1: predict done
    done_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(
        logits=done_logit, 
        labels=jnp.expand_dims(traj_batch.done, -1)
    ))
    
    # task 2: predict x= φ(s) - γ * φ(s')
    phi_next = network.apply(params, traj_batch.next_obs, method=network.phi)
    x_theta = phi_curr - config["GAMMA"] * phi_next

    # Update A to predict done and 
    forward_loss = jnp.mean((x_theta - x_hat) ** 2)
    
    # =========================================================
    # STAGE 2: Exogenous Grounding (Updates z_net and w)
    # =========================================================
    # Functional Firewall: Detach all g_A parameters in the PyTree
    # This is JIT-safe as it maps over the structure without explicit loops.
    def firewall_map(path, val):
        # path is a tuple of DictKey objects. 
        # Check if any part of the path contains 'g_A' modules.
        is_g_A = any('g_A' in str(p) for p in path)
        return jax.lax.stop_gradient(val) if is_g_A else val

    params_frozen_A = jax.tree_util.tree_map_with_path(firewall_map, params)

    # z is live. x_hat_freeze remains differentiable w.r.t z.
    x_raw_freeze, done_logit_freeze = network.apply(params_frozen_A, z, method=network.g_A)
    p_done_freeze = jax.nn.sigmoid(done_logit_freeze)

    # Expected X-hat based on g(z)'s prediction of x-hat and prediction of done.
    x_hat_freeze = p_done_freeze * jax.lax.stop_gradient(phi_curr)  \
                 + (1.0 - p_done_freeze) * x_raw_freeze
    
    # w is live. Predict immediate reward.
    reward_pred = network.apply(params, x_hat_freeze, method=network.w).squeeze(-1)
    reward_loss = jnp.mean((reward_pred - traj_batch.reward) ** 2)

    # Stage 3: VIC Reg.
    var_loss, cov_loss = vic_loss_fn(phi_curr)

    # Stage 4: Value Loss:
    v_loss = helpers.v_loss_fn(params, network, traj_batch, advantages, targets, config)
    
    a = config.get("VICREG_PHI_WEIGHT", 1.0)
    vic_loss = a * (var_loss + cov_loss)
    # =========================================================
    # PPO Actor Loss (Standard)
    # =========================================================
    loss_actor, entropy = helpers.pi_loss_fn(params, network, traj_batch, advantages, config)
    
    total_loss = forward_loss + done_loss + reward_loss + loss_actor - (config["ENT_COEF"] * entropy) + vic_loss + v_loss

    loss_dict = {   'forward_loss': forward_loss,
                    'done_loss': done_loss,
                    'reward_loss': reward_loss,
                    "vic_loss_var": var_loss, 
                    "vic_loss_cov": cov_loss,
                    "v_loss": v_loss
    }
    return total_loss, loss_dict

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

def make_train(config):

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    obs_shape = env.observation_space(env_params).shape

    def train(rng):
        k = config.get('k', 32)
        rng, init_rng = jax.random.split(rng)
        is_continuous = isinstance(env.action_space(env_params), spaces.Box)
        action_dim = env.action_space(env_params).shape[0] if is_continuous else env.action_space(env_params).n
        config['IS_CONTINUOUS'] = is_continuous
        network = IVActorCritic(action_dim=action_dim, is_continuous=False, z_dim=k, phi_dim=k)
        network_params = network.init(init_rng, jnp.zeros(obs_shape), jnp.zeros(action_dim))
        train_state = networks.initialize_flax_train_state(config, network, network_params,)
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info['real_next_obs']
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)

                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, info
                )
                
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # --- ADVANTAGE CALCULATION ---
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], config["GAE_LAMBDA"],)
            _, target = helpers.calculate_gae(traj_batch, config["GAMMA"], 1.0,)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(iv_loss_fn, has_aux=True)
                    (total_loss, loss_dict), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_dict

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                # loss_info is now a tuple of 4 arrays: (total_loss, value_loss, loss_actor, entropy)
                train_state, loss_dict = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), loss_dict

            initial_update_state = (train_state, traj_batch, advantages, target, rng)
            update_state, loss_dict = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state
            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            metric.update({
                k: v.mean() 
                for k, v in loss_dict.items() 
            })
            # Shared Metrics
            metric.update(
                {
                    "mean_rew": traj_batch.reward.mean(),
                }
            )
            # def value_metrics(evaluator, network, params, random_policy=False):
            value_metrics = bellman_error.value_metrics(evaluator, network, train_state.params, random_policy=False)
            metric.update(value_metrics)

            runner_state = (train_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
