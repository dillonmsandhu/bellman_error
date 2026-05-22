from core.imports import *
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
from flax.training.train_state import TrainState


class PQN_CNN(nn.Module):
    norm_type: str = "layer_norm"
    final_hidden_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # 1. Setup normalization
        if self.norm_type == "layer_norm":
            normalize = lambda tensor: nn.LayerNorm()(tensor)
        else:
            normalize = lambda tensor: tensor

        assert x.ndim in (3, 4), f"Input shape should be (H, W, C) or (B, H, W, C), got {x.shape}"
        
        # 2. Extract leading batch dimensions generically (CNNTorso style)
        # If 3D, batch_dims will be (1,) to preserve your existing init/unbatched code behavior.
        # If 4D, batch_dims will be (B,) matching the incoming environment batch.
        batch_dims = (x.shape[0],) if x.ndim == 4 else (1,)

        if x.ndim == 3:
            x = x[None, ...]
            
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=1,
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)

        # 4. Flatten using the dimension-agnostic torso style
        x = x.reshape(*batch_dims, -1)

        # 5. Final projection
        x = nn.Dense(
            self.final_hidden_dim, 
            kernel_init=orthogonal(jnp.sqrt(2)), 
            bias_init=constant(0.0)
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        
        return x

# class PQN_CNN(nn.Module):
#     norm_type: str = "layer_norm"
#     final_hidden_dim: int = 128

#     @nn.compact
#     def __call__(self, x: jnp.ndarray):
#         if self.norm_type == "layer_norm":
#             normalize = lambda x: nn.LayerNorm()(x)
#         else:
#             normalize = lambda x: x
#         assert x.ndim in (3,4), f"Input shape to channel-wise CNN should be (H, W, C) or (B, H, W, C), got shape {x.shape}"
#         if x.ndim == 3:  # Shape (H, W, C) -> Add batch dimension
#             x = x[None, ...]  # Shape becomes (1, H, W, C)
#         batch_conv = nn.vmap(nn.Conv, in_axes=0, out_axes=0, variable_axes={'params': None}, split_rngs={'params': None})
#         x = batch_conv(
#             16,
#             kernel_size=(3, 3),
#             strides=1,
#             padding="VALID",
#             kernel_init=nn.initializers.he_normal(),
#         )(x)
#         x = normalize(x)
#         x = nn.relu(x)
#         x = x.reshape((x.shape[0], -1))
#         x = nn.Dense(self.final_hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
#         x = normalize(x)
#         x = nn.relu(x)
#         return x

class PQN(nn.Module):
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = PQN_CNN(self.norm_type, self.final_hidden_dim)(x)
        x = nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros, bias_init = nn.initializers.zeros)(x)
        return x

class PolicyHead(nn.Module):
    action_dim: int
    is_continuous: bool = False

    @nn.compact
    def __call__(self, x):
        if not self.is_continuous:
            # Discrete: Output Logits
            logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
            return distrax.Categorical(logits=logits)
        else:
            # Continuous: Output Mean and Log Std
            loc = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            return distrax.MultivariateNormalDiag(loc=loc, scale_diag=jnp.exp(log_std))


class PQN_AC(nn.Module):
    """Actor critic with seperate small CNNs for the actor and critic"""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    is_continuous: bool = False
    
    def setup(self):
        self.actor_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.final_hidden_dim)
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.final_hidden_dim)
        self.actor_head = PolicyHead(self.action_dim, self.is_continuous)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init = constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns (pi(s), piV(s))"""
        v = self.value(x)
        pi = self.policy(x)
        return pi, v
    
    def policy(self, x: jnp.ndarray):
        """Returns pi(s)"""
        actor_features = self.actor_cnn(x)
        return self.actor_head(actor_features)

    def value_features(self, x: jnp.ndarray):
        """Returns V(s)"""
        critic_features = self.critic_cnn(x)
        return critic_features
    
    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        """Returns V(s)"""
        critic_features = self.value_features(x)
        return self.value_from_features(critic_features)
    
    def act(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        """Samples an action from the policy."""
        policy = self.policy(x)
        action = policy.sample(seed=key)
        return action.squeeze()

class PQN_Critic(nn.Module):
    """Actor critic with seperate small CNNs for the actor and critic"""
    action_dim: int
    final_hidden_dim: int
    norm_type: str = "none"
    
    def setup(self):
        self.critic_cnn = PQN_CNN(norm_type=self.norm_type, final_hidden_dim = self.final_hidden_dim)
        self.critic_head = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init = constant(0.0))

    def __call__(self, x: jnp.ndarray):
        """Returns V(s)"""
        v = self.value(x)
        return v
    
    def value_features(self, x: jnp.ndarray):
        critic_features = self.critic_cnn(x)
        return critic_features
    
    def value_from_features(self, phi):
        return self.critic_head(phi).squeeze(-1)

    def value(self, x: jnp.ndarray):
        """Returns V(s)"""
        return self.value_from_features(self.value_features(x))
    

def initialize_flax_train_state(config, network, params):
    # --- PPO Agent Scheduler & Optimizer ---
    total_grad_steps = config["NUM_UPDATES"] * config["NUM_MINIBATCHES"] * config["NUM_EPOCHS"]

    if config.get('OPTIMIZER','AdamW')=='AdamW':
        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=config["LR_END"],
            transition_steps=total_grad_steps
        )
        tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adamw(lr_scheduler, 
                weight_decay = config.get('WEIGHT_DECAY', 1e-2),
                eps=config.get('ADAM_EPS', 1e-5)
                ),
        )
        # tx = optax.chain(
        #     optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        #     optax.adamw(lr_scheduler, eps=1e-5),
        # )
    elif config.get('OPTIMIZER','AdamW')=='SGD':
        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"] * 1000,
            end_value=config["LR_END"] * 1000,
            transition_steps=total_grad_steps
        )
        tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.sgd(lr_scheduler),
        )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    return train_state

def initialize_network(rng, obs_shape, env, env_params, k, n_heads: int, layer_norm: bool):
    # Detect if continuous
    try:
        action_dim = env.action_space(env_params).n
        is_continuous = False 
    except:
        is_continuous = True 
        action_dim = env.action_space(env_params).shape[0]
    
    norm_type = 'layer_norm' if layer_norm else 'None'
    if n_heads == 2:
        model = PQN_AC(action_dim=action_dim, is_continuous=is_continuous, final_hidden_dim=k, norm_type= norm_type)
    elif n_heads == 1:
        model = PQN_Critic(action_dim=action_dim, final_hidden_dim=k, norm_type=norm_type)

    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, jnp.zeros(obs_shape))
    print('number of features is ', model.final_hidden_dim)
    return model, params

