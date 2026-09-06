"""MuZero with production KataGo trunk/heads and a spatial latent state."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from nanoalphazero.model import (
    KataGoTrunk, GenericPolicyHead, GoPolicyHead, ChessPolicyHead, ValueHead,
    resolve_preset, value_from_logits,
)


def normalize_spatial(x):
    lo = jnp.min(x, axis=(1, 2, 3), keepdims=True)
    span = jnp.max(x, axis=(1, 2, 3), keepdims=True) - lo
    return (x - lo) / jnp.maximum(span, 1e-5)


def action_planes(action, height, width, num_actions, env_id):
    """Encode the action definition, without a board or legality oracle."""
    batch = action.shape[0]
    if env_id == "connect_four":
        return jnp.broadcast_to(jax.nn.one_hot(action, width)[:, None, :, None],
                                (batch, height, width, 1))
    if env_id == "chess":
        planes = jax.nn.one_hot(action, num_actions).reshape(batch, height, width, 73)
        # Inverse of production ChessPolicyHead's final coordinate rotation.
        return jnp.rot90(planes, k=1, axes=(1, 2))
    board = jax.nn.one_hot(action, height * width).reshape(batch, height, width, 1)
    if env_id.startswith("go_"):
        passing = jnp.broadcast_to((action == height * width)[:, None, None, None], board.shape)
        return jnp.concatenate([board, passing.astype(board.dtype)], -1)
    if num_actions != height * width:
        raise ValueError("Unknown spatial action encoding")
    return board


class Prediction(nn.Module):
    num_actions: int
    env_id: str
    width: int
    depth: int
    activation: str

    @nn.compact
    def __call__(self, latent, return_wdl=False):
        cfg = resolve_preset(f"b{self.depth}c{self.width}nbt")
        mask = jnp.ones_like(latent[..., :1])
        mask_sum = jnp.sum(mask, axis=(1, 2))
        if self.env_id.startswith("go_"):
            head = GoPolicyHead(cfg["c_p1"], cfg["c_g1"], self.activation)
        elif self.env_id == "chess":
            head = ChessPolicyHead(cfg["c_p1"], cfg["c_g1"], self.activation)
        else:
            head = GenericPolicyHead(self.num_actions, cfg["c_p1"], self.activation)
        logits = head(latent, mask, mask_sum)
        value_logits = ValueHead(cfg["c_v1"], cfg["c_v2"], self.activation)(latent, mask, mask_sum)
        if return_wdl:
            return logits, value_from_logits(value_logits), jax.nn.softmax(value_logits, -1)
        return logits, value_from_logits(value_logits)


class SpatialMuZero(nn.Module):
    num_actions: int
    env_id: str
    width: int
    depth: int
    activation: str = "mish"
    use_rvgl: bool = True
    remat_blocks: bool = False

    def setup(self):
        cfg = resolve_preset(f"b{self.depth}c{self.width}nbt")
        trunk = {k: v for k, v in cfg.items() if k in (
            "c_trunk", "c_mid", "c_gpool", "block_gpool", "internal_length")}
        from nanoalphazero.research.muzero.remat import RematerializedTrunk
        trunk_type = RematerializedTrunk if self.remat_blocks else KataGoTrunk
        self.representation = trunk_type(**trunk, activation=self.activation, use_rvgl=self.use_rvgl)
        self.dynamics = trunk_type(**trunk, activation=self.activation, use_rvgl=self.use_rvgl)
        self.prediction = Prediction(self.num_actions, self.env_id, self.width, self.depth, self.activation)
        self.reward = ValueHead(cfg["c_v1"], cfg["c_v2"], self.activation)

    def initial(self, observation):
        latent, _, _ = self.representation(observation)
        latent = normalize_spatial(latent)
        logits, value = self.prediction(latent)
        return latent, logits, value

    def recurrent(self, latent, action):
        planes = action_planes(action, latent.shape[1], latent.shape[2], self.num_actions, self.env_id)
        features, mask, mask_sum = self.dynamics(jnp.concatenate([latent, planes], -1))
        reward = value_from_logits(self.reward(features, mask, mask_sum))
        latent = normalize_spatial(features)
        logits, value = self.prediction(latent)
        return latent, reward, logits, value

    def initial_with_wdl(self, observation):
        latent, _, _ = self.representation(observation)
        latent = normalize_spatial(latent)
        logits, value, probabilities = self.prediction(latent, return_wdl=True)
        return latent, logits, value, probabilities

    def __call__(self, observation, action):
        latent, _, _ = self.initial(observation)
        return self.recurrent(latent, action)
