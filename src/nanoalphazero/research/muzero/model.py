"""Small vector-latent MuZero baseline, with no observation reconstruction."""

import flax.linen as nn
import jax
import jax.numpy as jnp


def scale_gradient(x, scale):
    return scale * x + (1 - scale) * jax.lax.stop_gradient(x)


def normalize(x):
    lo = jnp.min(x, axis=-1, keepdims=True)
    span = jnp.max(x, axis=-1, keepdims=True) - lo
    return (x - lo) / jnp.maximum(span, 1e-5)


def build_model(config):
    if config.get("network", "vector") == "vector":
        return MuZero(config["num_actions"], config["width"], config["depth"])
    if config["network"] == "spatial":
        from nanoalphazero.research.muzero.spatial import SpatialMuZero
        return SpatialMuZero(config["num_actions"], config["env_id"], config["width"], config["depth"],
                             config["activation"], config["use_rvgl"])
    raise ValueError("Unknown MuZero network")


class Tower(nn.Module):
    width: int
    depth: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.width)(x))
        for _ in range(self.depth):
            residual = nn.relu(nn.Dense(self.width)(x))
            x = nn.relu(x + nn.Dense(self.width)(residual))
        return x


class MuZero(nn.Module):
    num_actions: int
    width: int = 128
    depth: int = 2

    def setup(self):
        self.representation = Tower(self.width, self.depth)
        self.dynamics = Tower(self.width, self.depth)
        self.prediction = Tower(self.width, self.depth)
        self.policy = nn.Dense(self.num_actions)
        self.value = nn.Dense(1)
        self.reward = nn.Dense(1)

    def predict(self, latent):
        features = self.prediction(latent)
        return self.policy(features), jnp.tanh(self.value(features)[..., 0])

    def initial(self, observation):
        flat = observation.astype(jnp.float32).reshape((observation.shape[0], -1))
        latent = normalize(self.representation(flat))
        logits, value = self.predict(latent)
        return latent, logits, value

    def recurrent(self, latent, action):
        action = jax.nn.one_hot(action, self.num_actions)
        features = self.dynamics(jnp.concatenate([latent, action], axis=-1))
        reward = jnp.tanh(self.reward(features)[..., 0])
        latent = normalize(features)
        logits, value = self.predict(latent)
        return latent, reward, logits, value

    def __call__(self, observation, action):
        latent, _, _ = self.initial(observation)
        return self.recurrent(latent, action)
