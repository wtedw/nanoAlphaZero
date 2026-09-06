"""Memory adapter composing the production trunk's unchanged building blocks."""

import math

import flax.linen as nn
import jax.numpy as jnp

from nanoalphazero.model import (
    KataGoTrunk, NestedBottleneckResBlock, NormMask, _ACTS, kata_init,
)


class RematerializedTrunk(KataGoTrunk):
    """Recompute each nested block during backward; preserve parameter paths.

    This mirrors only KataGoTrunk's composition because its block factory is
    not configurable. All layers, initializers and blocks remain production
    implementations. Explicit block names retain checkpoint/init equivalence.
    """

    @nn.compact
    def __call__(self, input_spatial, input_global=None, mask=None):
        input_spatial = input_spatial.astype(jnp.float32)
        if mask is None:
            mask = jnp.ones_like(input_spatial[..., :1])
        mask_sum = jnp.sum(mask, axis=(1, 2))
        out = nn.Conv(self.c_trunk, (3, 3), use_bias=False,
                      kernel_init=kata_init(0.8, self.activation))(input_spatial * mask)
        if input_global is not None:
            out = out + nn.Dense(self.c_trunk, use_bias=False,
                                 kernel_init=kata_init(0.6, self.activation))(
                                     input_global.astype(jnp.float32))[:, None, None, :]
        block = nn.remat(NestedBottleneckResBlock)
        fixup_scale = 1.0 / math.sqrt(len(self.block_gpool))
        for index, use_gpool in enumerate(self.block_gpool):
            out = out + block(
                self.c_trunk, self.c_mid, self.internal_length,
                fixup_scale, self.activation,
                c_gpool=self.c_gpool if use_gpool else None,
                use_rvgl=self.use_rvgl,
                name=f"NestedBottleneckResBlock_{index}",
            )(out, mask, mask_sum)
        out = _ACTS[self.activation](NormMask()(out, mask))
        return out, mask, mask_sum
