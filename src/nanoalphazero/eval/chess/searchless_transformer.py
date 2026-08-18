# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0. Modified for standalone,
# batched inference in nanoAlphaZero.
"""Minimal Searchless Chess transformer inference implementation."""

from __future__ import annotations

import dataclasses
import enum
import functools
from typing import NamedTuple

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np


class Predictor(NamedTuple):
    initial_params: object
    predict: object


class PositionalEncodings(enum.Enum):
    SINUSOID = enum.auto()
    LEARNED = enum.auto()


@dataclasses.dataclass(kw_only=True)
class TransformerConfig:
    seed: int = 1
    vocab_size: int
    output_size: int | None = None
    embedding_dim: int = 64
    num_layers: int = 4
    num_heads: int = 8
    use_causal_mask: bool = True
    emb_init_scale: float = 0.02
    pos_encodings: PositionalEncodings = PositionalEncodings.SINUSOID
    max_sequence_length: int | None = None
    widening_factor: int = 4
    apply_qk_layernorm: bool = False
    apply_post_ln: bool = True

    def __post_init__(self):
        if self.output_size is None:
            self.output_size = self.vocab_size


class MultiHeadDotProductAttention(hk.Module):
    def __init__(
        self,
        num_heads: int,
        num_hiddens_per_head: int,
        name: str | None = None,
        apply_qk_layernorm: bool = False,
    ) -> None:
        super().__init__(name=name)
        self._num_heads = num_heads
        self._num_hiddens_per_head = num_hiddens_per_head
        self._apply_qk_layernorm = apply_qk_layernorm

    def __call__(self, inputs_q, inputs_kv, mask=None):
        batch_size, sequence_length, embedding_size = inputs_q.shape
        num_hiddens = self._num_hiddens_per_head * self._num_heads
        q = hk.Linear(num_hiddens, with_bias=False)(inputs_q)
        k = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
        if self._apply_qk_layernorm:
            q = layer_norm(q)
            k = layer_norm(k)
        v = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
        new_shape = (
            batch_size,
            -1,
            self._num_heads,
            self._num_hiddens_per_head,
        )
        q = jnp.reshape(q, new_shape)
        k = jnp.reshape(k, new_shape)
        v = jnp.reshape(v, new_shape)
        attention = jnp.einsum("bthd,bThd->bhtT", q, k)
        attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)
        if mask is not None:
            attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)
        normalized_attention = jnn.softmax(attention)
        output = jnp.einsum("bhtT,bThd->bthd", normalized_attention, v)
        output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
        return hk.Linear(embedding_size, with_bias=False)(output)


def sinusoid_position_encoding(sequence_length: int, hidden_size: int, max_timescale=1e4):
    freqs = np.arange(0, hidden_size + 1, 2)
    inv_freq = max_timescale ** (-freqs / hidden_size)
    pos_seq = np.arange(start=0, stop=sequence_length)
    sinusoid_inp = np.einsum("i,j->ij", pos_seq, inv_freq)
    embeddings = np.concatenate(
        [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
    )
    return embeddings[:, :hidden_size]


def embed_sequences(sequences, config: TransformerConfig):
    embs_init = hk.initializers.TruncatedNormal(stddev=config.emb_init_scale)
    embeddings_layer = hk.Embed(
        vocab_size=config.vocab_size,
        embed_dim=config.embedding_dim,
        lookup_style=hk.EmbedLookupStyle.ARRAY_INDEX,
        w_init=embs_init,
    )
    embeddings = embeddings_layer(sequences)
    embeddings *= jnp.sqrt(config.embedding_dim)
    _, sequence_length, embedding_size = embeddings.shape
    if config.pos_encodings == PositionalEncodings.SINUSOID:
        pos_encodings = sinusoid_position_encoding(sequence_length, embedding_size)
    else:
        assert sequence_length <= config.max_sequence_length
        positions = jnp.arange(sequence_length)
        pos_encodings = hk.Embed(
            vocab_size=config.max_sequence_length, embed_dim=embedding_size
        )(positions)
    return embeddings + pos_encodings


def layer_norm(x):
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def shift_right(sequences):
    bos_array = jnp.zeros((sequences.shape[0], 1), dtype=jnp.uint8)
    padded_sequences = jnp.concatenate([bos_array, sequences], axis=1)
    return padded_sequences[:, :-1]


def _mlp_block(inputs, config: TransformerConfig):
    ffn_dim = config.embedding_dim * config.widening_factor
    split_1 = hk.Linear(ffn_dim, with_bias=False)(inputs)
    split_2 = hk.Linear(ffn_dim, with_bias=False)(inputs)
    gate_output = jnn.silu(split_1) * split_2
    return hk.Linear(config.embedding_dim, with_bias=False)(gate_output)


def _attention_block(inputs, config: TransformerConfig):
    batch_size, sequence_length = inputs.shape[:2]
    causal_mask = (
        np.tril(np.ones((batch_size, 1, sequence_length, sequence_length)))
        if config.use_causal_mask
        else None
    )
    block = MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        num_hiddens_per_head=config.embedding_dim // config.num_heads,
        apply_qk_layernorm=config.apply_qk_layernorm,
    )
    return block(inputs_q=inputs, inputs_kv=inputs, mask=causal_mask)


def transformer_decoder(targets, config: TransformerConfig):
    inputs = shift_right(targets)
    embeddings = embed_sequences(inputs, config)
    h = embeddings
    for _ in range(config.num_layers):
        attention_input = layer_norm(h)
        h += _attention_block(attention_input, config)
        mlp_input = layer_norm(h)
        h += _mlp_block(mlp_input, config)
    if config.apply_post_ln:
        h = layer_norm(h)
    logits = hk.Linear(config.output_size)(h)
    return jnn.log_softmax(logits, axis=-1)


def build_transformer_predictor(config: TransformerConfig) -> Predictor:
    model = hk.transform(functools.partial(transformer_decoder, config=config))
    return Predictor(initial_params=model.init, predict=model.apply)
