"""KataGo-style neural-network model definitions."""

import math
import re
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Neural network model
#
# KataGo's fixup nested-bottleneck architecture is the sole network used by
# nanoAlphaZero. The trunk is game-agnostic. Go uses KataGo's native board-point
# plus pass policy head; other games use a generic action-space policy head.
# MCTS consumes the scalar P(win)-P(loss), while training uses the raw WDL logits.
# =============================================================================
_TRUNC_STD_CORRECTION = 0.87962566103423978

_GAINS = {
    "relu": math.sqrt(2.0),
    "elu": math.sqrt(1.55052),
    "mish": math.sqrt(2.210277),
    "silu": math.sqrt(2.0),
    "gelu": math.sqrt(2.351718),
    "hardswish": math.sqrt(2.0),
    "identity": 1.0,
}


def _mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


_ACTS = {
    "relu": jax.nn.relu,
    "elu": jax.nn.elu,
    "mish": _mish,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "hardswish": jax.nn.hard_swish,
    "identity": lambda x: x,
}


def kata_init(scale: float, activation: str, fan_in: Optional[int] = None):
    """KataGo's ``init_weights``: trunc normal, std = scale*gain/sqrt(fan_in).

    ``fan_in`` defaults to prod(shape[:-1]), which matches torch's fan-in for
    both HWIO conv kernels (kh*kw*c_in) and (in, out) dense kernels. Pass it
    explicitly for biases (KataGo's ``fan_tensor=weight``).
    """
    gain = _GAINS[activation]

    def init(key, shape, dtype=jnp.float32):
        fi = fan_in if fan_in is not None else int(np.prod(shape[:-1]))
        std = scale * gain / math.sqrt(fi) / _TRUNC_STD_CORRECTION
        if std < 1e-10:
            return jnp.zeros(shape, dtype)
        return std * jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype)

    return init


class NormMask(nn.Module):
    """KataGo's NormMask at norm_kind="fixup": ``(x*(1+gamma) + beta) * mask``.

    ``use_gamma`` corresponds to fixup_use_gamma=True (the second NormActConv
    of each ResBlock and the closing 1x1 of each nested block). All base
    configs set gamma_weight_decay_center_1, so gamma is stored centered at 0
    and applied as (gamma + 1). Also covers BiasMask (use_gamma=False).
    """

    use_gamma: bool = False

    @nn.compact
    def __call__(self, x, mask):
        c = x.shape[-1]
        beta = self.param("beta", nn.initializers.zeros, (c,))
        if self.use_gamma:
            gamma = self.param("gamma", nn.initializers.zeros, (c,))
            x = x * (gamma + 1.0)
        return (x + beta) * mask


def kata_gpool(x, mask, mask_sum_hw):
    """KataGPool: masked mean, size-scaled mean, masked max. [N,H,W,C] -> [N,3C].

    Assumes off-board activations are exactly 0 (guaranteed by NormMask) and
    that the activation maps 0 -> 0 and is > -1, so ``x + (mask - 1)`` makes
    off-board positions lose the max.
    """
    layer_mean = jnp.sum(x, axis=(1, 2)) / mask_sum_hw  # [N, C]
    s = jnp.sqrt(mask_sum_hw) - 14.0  # [N, 1]
    layer_max = jnp.max(x + (mask - 1.0), axis=(1, 2))  # [N, C]
    return jnp.concatenate([layer_mean, layer_mean * (s / 10.0), layer_max], axis=1)


def kata_value_head_gpool(x, mask, mask_sum_hw):
    """KataValueHeadGPool: mean with linear and quadratic board-size features."""
    layer_mean = jnp.sum(x, axis=(1, 2)) / mask_sum_hw
    s = jnp.sqrt(mask_sum_hw) - 14.0
    return jnp.concatenate(
        [
            layer_mean,
            layer_mean * (s / 10.0),
            layer_mean * (jnp.square(s) / 100.0 - 0.1),
        ],
        axis=1,
    )


class KataConvAndGPool(nn.Module):
    """Regular 3x3 conv branch + gpool branch feeding a bias back in."""

    c_out: int
    c_gpool: int
    scale: float  # fixup init scale for this position in the net
    activation: str

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        act = _ACTS[self.activation]
        # Fixup branch of KataConvAndGPool.initialize: r_scale=0.8 on the
        # regular conv, sqrt(scale)*sqrt(0.6) on both halves of the g branch.
        outr = nn.Conv(
            self.c_out, (3, 3), use_bias=False,
            kernel_init=kata_init(self.scale * 0.8, self.activation),
        )(x)
        g_scale = math.sqrt(self.scale) * math.sqrt(0.6)
        outg = nn.Conv(
            self.c_gpool, (3, 3), use_bias=False,
            kernel_init=kata_init(g_scale, self.activation),
        )(x)
        outg = NormMask()(outg, mask)
        outg = act(outg)
        outg = kata_gpool(outg, mask, mask_sum_hw)  # [N, 3*c_gpool]
        outg = nn.Dense(
            self.c_out, use_bias=False,
            kernel_init=kata_init(g_scale, self.activation),
        )(outg)
        return outr + outg[:, None, None, :]


class RepVGGLinearConv(nn.Module):
    """RVGL's separate 3x3/1x1 parameters evaluated as one fused convolution."""

    c_out: int
    kernel_size: int
    scale: float
    activation: str

    @nn.compact
    def __call__(self, x):
        c_in = x.shape[-1]
        kernel3 = self.param(
            "kernel_3x3",
            kata_init(self.scale * 0.8, self.activation),
            (self.kernel_size, self.kernel_size, c_in, self.c_out),
        )
        kernel1 = self.param(
            "kernel_1x1",
            kata_init(self.scale * 0.6, self.activation),
            (1, 1, c_in, self.c_out),
        )
        center = self.kernel_size // 2
        combined_kernel = kernel3.at[center, center].add(kernel1[0, 0])
        combined_kernel = combined_kernel.astype(x.dtype)
        return jax.lax.conv_general_dilated(
            lhs=x,
            rhs=combined_kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )


class NormActConv(nn.Module):
    """norm -> act -> conv (or conv+gpool)."""

    c_out: int
    kernel_size: int
    scale: float
    activation: str
    c_gpool: Optional[int] = None
    use_gamma: bool = False  # fixup_use_gamma
    use_rvgl: bool = True

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        out = NormMask(use_gamma=self.use_gamma)(x, mask)
        out = _ACTS[self.activation](out)
        if self.c_gpool is not None:
            return KataConvAndGPool(
                self.c_out, self.c_gpool, self.scale, self.activation
            )(out, mask, mask_sum_hw)
        if self.use_rvgl and self.kernel_size > 1:
            # KataGo -rvgl: parallel linear 3x3 and 1x1 branches. Their
            # initialization variances add to one (0.8**2 + 0.6**2 == 1).
            # Since there is no nonlinearity between the branches, the 1x1
            # kernel is folded into the 3x3 for the actual convolution while
            # remaining a separate parameter with its own optimizer state.
            return RepVGGLinearConv(
                self.c_out, self.kernel_size, self.scale, self.activation
            )(out)
        return nn.Conv(
            self.c_out, (self.kernel_size, self.kernel_size), use_bias=False,
            kernel_init=kata_init(self.scale, self.activation),
        )(out)


class ResBlock(nn.Module):
    """Inner two-conv residual block. Returns the residual only."""

    c_main: int
    c_mid: int
    fixup_scale: float
    activation: str
    c_gpool: Optional[int] = None
    use_rvgl: bool = True

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        c1_out = self.c_mid - (self.c_gpool or 0)
        out = NormActConv(
            c1_out, 3, self.fixup_scale, self.activation,
            c_gpool=self.c_gpool, use_rvgl=self.use_rvgl,
        )(x, mask, mask_sum_hw)
        # Fixup: second conv zero-init, and its NormMask carries a gamma.
        out = NormActConv(
            self.c_main, 3, 0.0, self.activation,
            use_gamma=True, use_rvgl=self.use_rvgl,
        )(out, mask, mask_sum_hw)
        return out


class NestedBottleneckResBlock(nn.Module):
    """KataGo "bottlenest{L}" block: 1x1 down, L residual ResBlocks, 1x1 up.

    Returns the residual only; the caller adds it to the trunk.
    """

    c_trunk: int
    c_mid: int
    internal_length: int
    fixup_scale: float
    activation: str
    c_gpool: Optional[int] = None
    use_rvgl: bool = True

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        inner_scale = self.fixup_scale ** (1.0 / (1.0 + self.internal_length))
        out = NormActConv(self.c_mid, 1, inner_scale, self.activation)(
            x, mask, mask_sum_hw
        )
        for i in range(self.internal_length):
            out = out + ResBlock(
                self.c_mid, self.c_mid, inner_scale, self.activation,
                c_gpool=(self.c_gpool if i == 0 else None),
                use_rvgl=self.use_rvgl,
            )(out, mask, mask_sum_hw)
        out = NormActConv(
            self.c_trunk, 1, 0.0, self.activation, use_gamma=True
        )(out, mask, mask_sum_hw)
        return out


class GoPolicyHead(nn.Module):
    """KataGo policy head (version >= 15 pass pathway), single policy output.

    Returns [N, H*W + 1] logits; the last entry is the pass move. Off-board
    positions get logit - 5000 so they vanish after softmax.
    """

    c_p1: int
    c_g1: int
    activation: str

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        act = _ACTS[self.activation]
        outp = nn.Conv(
            self.c_p1, (1, 1), use_bias=False,
            kernel_init=kata_init(0.8, self.activation),
        )(x)
        outg = nn.Conv(
            self.c_g1, (1, 1), use_bias=False,
            kernel_init=kata_init(1.0, self.activation),
        )(x)
        outg = NormMask()(outg, mask)  # BiasMask
        outg = act(outg)
        outg = kata_gpool(outg, mask, mask_sum_hw)  # [N, 3*c_g1]

        outpass = nn.Dense(
            self.c_p1, use_bias=True,
            kernel_init=kata_init(1.0, self.activation),
            bias_init=kata_init(0.2, self.activation, fan_in=3 * self.c_g1),
        )(outg)
        outpass = act(outpass)
        outpass = nn.Dense(
            1, use_bias=False, kernel_init=kata_init(0.3, "identity")
        )(outpass)  # [N, 1]

        outg = nn.Dense(
            self.c_p1, use_bias=False,
            kernel_init=kata_init(0.6, self.activation),
        )(outg)
        outp = outp + outg[:, None, None, :]
        outp = NormMask()(outp, mask)  # bias2
        outp = act(outp)
        outp = nn.Conv(
            1, (1, 1), use_bias=False, kernel_init=kata_init(0.3, "identity")
        )(outp)  # [N, H, W, 1]
        outp = outp - (1.0 - mask) * 5000.0
        n = outp.shape[0]
        return jnp.concatenate([outp.reshape(n, -1), outpass], axis=1)


class GenericPolicyHead(nn.Module):
    """A plain action-space policy head over the trunk features for non-Go games.

    This is used based on explicit game identity, not action-space shape
    (e.g. Connect4's column actions). This head has no pass-pathway/gpool
    machinery, just conv -> norm -> act -> flatten -> Dense(action_space).
    """

    action_space: int
    c_p1: int
    activation: str

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        act = _ACTS[self.activation]
        p = nn.Conv(
            self.c_p1, (1, 1), use_bias=False,
            kernel_init=kata_init(0.8, self.activation),
        )(x)
        p = NormMask()(p, mask)
        p = act(p)
        p = p.reshape(p.shape[0], -1)
        return nn.Dense(
            self.action_space,
            kernel_init=nn.initializers.normal(stddev=1e-2),
            bias_init=nn.initializers.zeros,
        )(p)


class ChessPolicyHead(nn.Module):
    """KataGo-style policy head for pgx chess's 64x73 action encoding."""

    c_p1: int
    c_g1: int
    activation: str
    num_planes: int = 73

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        act = _ACTS[self.activation]
        outp = nn.Conv(
            self.c_p1,
            (1, 1),
            use_bias=False,
            kernel_init=kata_init(0.8, self.activation),
        )(x)
        outg = nn.Conv(
            self.c_g1,
            (1, 1),
            use_bias=False,
            kernel_init=kata_init(1.0, self.activation),
        )(x)
        outg = NormMask()(outg, mask)
        outg = act(outg)
        outg = kata_gpool(outg, mask, mask_sum_hw)
        outg = nn.Dense(
            self.c_p1,
            use_bias=False,
            kernel_init=kata_init(0.6, self.activation),
        )(outg)
        outp = outp + outg[:, None, None, :]
        outp = NormMask()(outp, mask)
        outp = act(outp)
        outp = nn.Conv(
            self.num_planes,
            (1, 1),
            use_bias=False,
            kernel_init=kata_init(0.3, "identity"),
        )(outp)
        # pgx observes chess after a 90-degree rotation. Undo it so row-major
        # flattening matches action = from_square * 73 + move_plane.
        outp = jnp.rot90(outp, k=-1, axes=(1, 2))
        return outp.reshape(outp.shape[0], -1)


class ValueHead(nn.Module):
    """KataGo value head, main 3-way {win, loss, noresult} logits only."""

    c_v1: int
    c_v2: int
    activation: str

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        act = _ACTS[self.activation]
        v1 = nn.Conv(
            self.c_v1, (1, 1), use_bias=False,
            kernel_init=kata_init(1.0, self.activation),
        )(x)
        v1 = NormMask()(v1, mask)  # bias1
        v1 = act(v1)
        pooled = kata_value_head_gpool(v1, mask, mask_sum_hw)  # [N, 3*c_v1]
        v2 = nn.Dense(
            self.c_v2, use_bias=True,
            kernel_init=kata_init(1.0, self.activation),
            bias_init=kata_init(0.2, self.activation, fan_in=3 * self.c_v1),
        )(pooled)
        v2 = act(v2)
        return nn.Dense(
            3, use_bias=True,
            kernel_init=kata_init(1.0, "identity"),
            bias_init=kata_init(0.2, "identity", fan_in=self.c_v2),
        )(v2)


class KataGoTrunk(nn.Module):
    """The bXcYnbt trunk only (input conv + nested-bottleneck blocks + final
    norm/act), shared by the native Go head and the generic action-space head.

    ``block_gpool[i]`` says whether trunk block i is the "gpool" flavor.
    Defaults are b10c384nbt. Inputs are NHWC; off-board input features should
    be zero (they are re-masked defensively anyway).
    """

    c_trunk: int = 384
    c_mid: int = 192
    c_gpool: int = 64
    block_gpool: Sequence[bool] = (
        False, False, True, False, False, True, False, False, True, False,
    )
    internal_length: int = 2
    activation: str = "relu"
    use_rvgl: bool = True

    @nn.compact
    def __call__(
        self,
        input_spatial,  # [N, H, W, C_spatial]
        input_global=None,  # [N, C_global] or None
        mask=None,  # [N, H, W, 1] on-board mask, or None for full board
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        input_spatial = input_spatial.astype(jnp.float32)
        if mask is None:
            mask = jnp.ones_like(input_spatial[..., :1])
        mask_sum_hw = jnp.sum(mask, axis=(1, 2))  # [N, 1]

        out = nn.Conv(
            self.c_trunk, (3, 3), use_bias=False,
            kernel_init=kata_init(0.8, self.activation),
        )(input_spatial * mask)
        if input_global is not None:
            out = out + nn.Dense(
                self.c_trunk, use_bias=False,
                kernel_init=kata_init(0.6, self.activation),
            )(input_global.astype(jnp.float32))[:, None, None, :]

        fixup_scale = 1.0 / math.sqrt(len(self.block_gpool))
        for use_gpool in self.block_gpool:
            out = out + NestedBottleneckResBlock(
                self.c_trunk, self.c_mid, self.internal_length,
                fixup_scale, self.activation,
                c_gpool=(self.c_gpool if use_gpool else None),
                use_rvgl=self.use_rvgl,
            )(out, mask, mask_sum_hw)

        out = NormMask()(out, mask)  # norm_trunkfinal (fixup: bias+mask)
        out = _ACTS[self.activation](out)
        return out, mask, mask_sum_hw


def value_from_logits(value_logits: jax.Array) -> jax.Array:
    """[N, 3] {win, loss, noresult} logits -> scalar value in [-1, 1] ([N])."""
    probs = jax.nn.softmax(value_logits, axis=-1)
    return probs[..., 0] - probs[..., 1]


def _gpool_every_3(n: int) -> Tuple[bool, ...]:
    """KataGo's nbt configs put the gpool flavor on blocks 3, 6, 9, ... except
    that a final block is never gpool (b18c384nbt stops at 15)."""
    return tuple((i % 3 == 0) and (i != n) for i in range(1, n + 1))


# Trunk/head sizes lifted from modelconfigs.py (fixup base configs).
PRESETS = {
    "b5c192nbt": dict(
        c_trunk=192, c_mid=96, c_gpool=32,
        block_gpool=(False, True, False, True, False),
        c_p1=32, c_g1=32, c_v1=32, c_v2=80,
    ),
    "b10c256nbt": dict(
        c_trunk=256, c_mid=128, c_gpool=64,
        block_gpool=_gpool_every_3(10),
        c_p1=32, c_g1=32, c_v1=32, c_v2=96,
    ),
    "b10c384nbt": dict(
        c_trunk=384, c_mid=192, c_gpool=64,
        block_gpool=_gpool_every_3(10),
        c_p1=48, c_g1=48, c_v1=48, c_v2=112,
    ),
    "b18c384nbt": dict(
        c_trunk=384, c_mid=192, c_gpool=64,
        block_gpool=_gpool_every_3(18),
        c_p1=48, c_g1=48, c_v1=96, c_v2=128,
    ),
    "b28c512nbt": dict(
        c_trunk=512, c_mid=256, c_gpool=64,
        block_gpool=_gpool_every_3(28),
        c_p1=64, c_g1=64, c_v1=128, c_v2=144,
    ),
}


_DYNAMIC_PRESET_RE = re.compile(r"^b(?P<blocks>[1-9]\d*)c(?P<channels>[1-9]\d*)(?:i(?P<internal>[1-9]\d*))?nbt$")


def resolve_preset(preset: str) -> dict:
    """Resolve a fixed or dynamic KataGo-style preset string.

    Fixed presets are lifted from KataGo modelconfigs.py. Dynamic presets use
    ``b{blocks}c{channels}nbt`` and are intentionally explicit so sweeps can
    target tiny models without adding many hard-coded names. ``i{internal}``
    may be inserted before ``nbt`` to override the nested-bottleneck internal
    length; otherwise KataGo's nbt default of 2 is used.
    """
    if preset in PRESETS:
        return dict(PRESETS[preset])

    match = _DYNAMIC_PRESET_RE.match(preset)
    if not match:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"Unknown KataGo preset {preset!r}. Use a fixed preset ({valid}) "
            "or dynamic form b{blocks}c{channels}nbt, e.g. b4c16nbt."
        )

    blocks = int(match.group("blocks"))
    channels = int(match.group("channels"))
    internal = int(match.group("internal") or 2)
    head_channels = max(1, channels // 8)
    return dict(
        c_trunk=channels,
        c_mid=max(1, channels // 2),
        c_gpool=max(1, channels // 8),
        block_gpool=_gpool_every_3(blocks),
        internal_length=internal,
        c_p1=head_channels,
        c_g1=head_channels,
        c_v1=head_channels,
        c_v2=max(4, channels // 4),
    )


class KataModel(nn.Module):
    """KataGo-style network matching nanoAlphaZero's model interface:

        masked_logits, value = model(obs, valid)

    obs is a pgx NHWC observation, valid a [B, action_space] legal-move mask,
    value a scalar in [-1, 1] from the current player's perspective.
    ``deterministic`` is accepted for signature compatibility (no dropout).

    CHESS ADAPTATION: originally this always used KataGo's native Go head,
    which only ever emits H*W + 1 logits (a Go board point + one pooled pass
    move) -- fine for go, but incompatible with chess's 4672-way
    from/to/promotion move encoding. The trunk itself was never Go-specific
    (just conv + nested-bottleneck blocks over an [N,H,W,C] tensor), so it's
    now factored out into KataGoTrunk and shared by game-specific heads:
    KataGo's own GoPolicyHead for Go, a spatial 73-plane head for chess, or a
    plain flatten+Dense GenericPolicyHead otherwise. Game identity is explicit
    so an unrelated game with H*W+1 actions is not mistaken for Go.
    """

    action_space: int
    is_go: bool
    is_chess: bool = False
    c_trunk: int = 384
    c_mid: int = 192
    c_gpool: int = 64
    block_gpool: Sequence[bool] = (
        False, False, True, False, False, True, False, False, True, False,
    )
    internal_length: int = 2
    c_p1: int = 48
    c_g1: int = 48
    c_v1: int = 48
    c_v2: int = 112
    activation: str = "relu"
    use_rvgl: bool = True

    @nn.compact
    def __call__(
        self,
        obs,
        valid,
        deterministic: bool = False,
        return_wdl_logits: bool = False,
    ):
        trunk = KataGoTrunk(
            c_trunk=self.c_trunk, c_mid=self.c_mid, c_gpool=self.c_gpool,
            block_gpool=self.block_gpool, internal_length=self.internal_length,
            activation=self.activation, use_rvgl=self.use_rvgl,
        )
        out, mask, mask_sum_hw = trunk(obs)

        if self.is_go and self.is_chess:
            raise ValueError("is_go and is_chess are mutually exclusive")
        if self.is_go:
            expected_actions = out.shape[1] * out.shape[2] + 1
            if self.action_space != expected_actions:
                raise ValueError(
                    "KataGo's Go policy head requires action_space == H*W+1; "
                    f"got action_space={self.action_space}, H*W+1={expected_actions}"
                )
            # Go: board points + pass, KataGo's own two-pathway policy head.
            # Keep the historical Flax path so existing Go checkpoints load.
            policy_logits = GoPolicyHead(
                self.c_p1, self.c_g1, self.activation, name="PolicyHead_0"
            )(out, mask, mask_sum_hw)
        elif self.is_chess:
            head = ChessPolicyHead(self.c_p1, self.c_g1, self.activation)
            expected_actions = out.shape[1] * out.shape[2] * head.num_planes
            if self.action_space != expected_actions:
                raise ValueError(
                    "ChessPolicyHead requires action_space == H*W*73; "
                    f"got action_space={self.action_space}, "
                    f"H*W*73={expected_actions}"
                )
            policy_logits = head(out, mask, mask_sum_hw)
        else:
            # Other non-Go action encodings.
            policy_logits = GenericPolicyHead(
                self.action_space, self.c_p1, self.activation
            )(out, mask, mask_sum_hw)

        value_logits = ValueHead(self.c_v1, self.c_v2, self.activation)(
            out, mask, mask_sum_hw
        )
        masked_logits = jnp.where(
            valid, policy_logits, jnp.finfo(policy_logits.dtype).min
        )
        value = value_from_logits(value_logits)
        if return_wdl_logits:
            return masked_logits, value, value_logits
        return masked_logits, value

    def sample(self, logits, key, test: bool = False):
        return (
            jnp.argmax(logits, axis=-1) if test else jax.random.categorical(key, logits)
        )

def make_model(config, rng, sharding=None):
    """Build the KataGo-style network using the existing width/depth settings."""
    preset = config.get(
        "katago_preset",
        f"b{config['conv_depth']}c{config['conv_width']}nbt",
    )
    env_id = str(config["env_id"])
    model_kwargs = resolve_preset(preset)
    model = KataModel(
        action_space=config["game_num_actions"],
        is_go=env_id.startswith("go_"),
        is_chess=env_id == "chess",
        **model_kwargs,
        activation=config.get("katago_activation", "mish"),
        use_rvgl=config.get("katago_use_rvgl", True),
    )
    observation = jnp.zeros((1,) + config["game_obs_shape"])
    valid_action_mask = jnp.ones((1, config["game_num_actions"]), dtype=bool)
    model_state = init_and_shard_model(
        config, model, rng, observation, valid_action_mask, sharding
    )
    return model, model_state


# Shard model params as REPLICATED across the mesh when enabled.
def init_and_shard_model(config, model, rng, obs, valid_mask, sharding):
    use_bf16 = config.get("use_bf16", False)

    def _init_fn(rng, obs, mask):
        variables = model.init(rng, obs, mask)
        if use_bf16:
            variables = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.bfloat16), variables
            )
        return variables

    if not config.get("enable_sharding", False) or sharding is None:
        return jax.jit(_init_fn)(rng, obs, valid_mask)

    abstract_variables = jax.eval_shape(_init_fn, rng, obs, valid_mask)
    sharding_tree = jax.tree_util.tree_map(lambda _: sharding, abstract_variables)
    sharded_init = jax.jit(_init_fn, out_shardings=sharding_tree)
    with sharding.mesh:
        model_state = sharded_init(rng, obs, valid_mask)
    return model_state

