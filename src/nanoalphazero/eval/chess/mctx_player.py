"""Safetensors KataModel player backed by the external MCTX opt search."""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mctx
import numpy as np
from pgx1.experimental import auto_reset

from nanoalphazero import core
from nanoalphazero.checkpoint import apply_checkpoint_model_config, load_checkpoint
from nanoalphazero.config import get_chess_config
from nanoalphazero.eval.chess.assets import sha256_file


_MODEL_CACHE: dict[tuple[str, str], tuple[Any, dict[str, Any], Any]] = {}


def _unpack_bitmask_vmap(bitmask):
    from pgx1.chess import unpack_bitmask

    return jax.vmap(unpack_bitmask)(bitmask)


def make_pgx1_env():
    """Build distinct normal-step and auto-reset search transitions."""
    import pgx1

    env = pgx1.make("chess", use_bitmask=True, return_observation=False)
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    autostep = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    observe = jax.jit(jax.vmap(env.observe))

    def init_dummy_estate(batch_size: int):
        return init(jax.random.split(jax.random.PRNGKey(0), batch_size))

    single = env.init(jax.random.PRNGKey(0))
    sample = init_dummy_estate(1)
    observation = observe(sample, sample.current_player)
    wrapped = core.WrappedEnv(
        obs_shape=tuple(observation.shape[1:]),
        num_actions=env.num_actions,
        init=init,
        step=step,
        autostep=autostep,
        init_dummy_estate=init_dummy_estate,
        single_estate=single,
        observe=observe,
        replay_batch=getattr(env, "replay_batch", None),
    )
    wrapped.data_sharding = core.DATA_PARALLEL_SHARDING
    return wrapped


def _validate_params(actual, expected, checkpoint: Path) -> None:
    actual_shapes = jax.tree.map(lambda value: tuple(value.shape), actual)
    expected_shapes = jax.tree.map(lambda value: tuple(value.shape), expected)
    if jax.tree.structure(actual_shapes) != jax.tree.structure(expected_shapes):
        raise ValueError(f"checkpoint parameter tree differs from model: {checkpoint}")
    mismatches = [
        (index, got, want)
        for index, (got, want) in enumerate(
            zip(
                jax.tree.leaves(actual_shapes),
                jax.tree.leaves(expected_shapes),
                strict=True,
            )
        )
        if got != want
    ]
    if mismatches:
        raise ValueError(
            f"checkpoint parameter shapes differ from model: {mismatches[:8]}"
        )


def resolved_mctx_config(agent: dict[str, Any], env, max_plies: int) -> dict[str, Any]:
    """Resolve checkpoint-independent search settings without hidden defaults."""
    search = dict(agent["search"])
    config = get_chess_config()
    config.update(
        {
            "env_id": "chess",
            "game_obs_shape": env.obs_shape,
            "game_num_actions": env.num_actions,
            "game_max_steps": int(max_plies),
            "enable_sharding": True,
            "mcts_bnk_rehydrate_fields": False,
            "mcts_return_search_tree": False,
            "mcts_return_summary": False,
            "mcts_use_advantage_weights": False,
            "mcts_advantage_scale": 1.0,
            "mcts_use_puct_interior": False,
            "exp_use_root_temperature": False,
            "exp_root_temperature": 1.5,
            "exp_use_all_temp": False,
            "exp_all_temp": 1.0,
        }
    )
    config.update(search)
    return config


def build_qtransforms(config: dict[str, Any]):
    """Construct root, final, and interior transforms from one specification."""
    common = {
        "rescale_values": bool(config["mcts_rescale_values"]),
        "value_scale": float(config["mcts_value_scale"]),
        "maxvisit_init": float(config["mcts_maxvisit_init"]),
        "use_mixed_value": bool(config["mcts_use_mixed_value"]),
    }
    root_spec = {
        **common,
        "visit_exponent": float(config["mcts_visit_exponent"]),
        "visit_aggregator": str(config["mcts_visit_aggregator"]),
    }
    interior_spec = {
        **common,
        "visit_exponent": float(
            config.get(
                "mcts_visit_exponent_interior",
                config["mcts_visit_exponent"],
            )
        ),
        "visit_aggregator": str(
            config.get(
                "mcts_visit_aggregator_interior",
                config["mcts_visit_aggregator"],
            )
        ),
    }
    root = functools.partial(mctx.qtransform_completed_by_mix_value, **root_spec)
    final = functools.partial(
        mctx.qtransform_completed_by_mix_value,
        **root_spec,
        return_extras=True,
    )
    interior = functools.partial(
        mctx.qtransform_completed_by_mix_value, **interior_spec
    )
    return root, final, interior, {"root": root_spec, "interior": interior_spec}


def make_eval_mcts(config: dict[str, Any], env, model):
    """Create the JIT MCTX opt call used for resident tournament batches."""
    qtransform, final_qtransform, interior_qtransform, qtransform_config = (
        build_qtransforms(config)
    )
    sharding = core.DATA_PARALLEL_SHARDING

    def root_fn(params, env_state):
        observation = env.observe(env_state, env_state.current_player)
        legal = _unpack_bitmask_vmap(env_state.legal_action_bitmask)
        observation = jax.lax.with_sharding_constraint(observation, sharding)
        legal = jax.lax.with_sharding_constraint(legal, sharding)
        prior_logits, value = model.apply({"params": params}, observation, legal)
        if config.get("exp_use_root_temperature", False):
            tau = float(config["exp_root_temperature"])
            prior_logits = jnp.where(
                legal, prior_logits / tau, jnp.finfo(prior_logits.dtype).min
            )
        if config.get("exp_use_all_temp", False):
            tau = float(config["exp_all_temp"])
            prior_logits = jnp.where(
                legal, prior_logits / tau, jnp.finfo(prior_logits.dtype).min
            )
        return mctx.RootFnOutput(
            prior_logits=prior_logits,
            value=value,
            embedding=env_state,
        )

    def recurrent_fn(params, rng_key, action, env_state):
        action = jnp.asarray(action, dtype=jnp.int32)
        previous_player = env_state.current_player
        previous_player = jax.lax.with_sharding_constraint(
            previous_player, sharding
        )
        action = jax.lax.with_sharding_constraint(action, sharding)
        batch_size = action.shape[0]
        env_state = env.autostep(
            env_state, action, jax.random.split(rng_key, batch_size)
        )
        observation = env.observe(env_state, env_state.current_player)
        legal = _unpack_bitmask_vmap(env_state.legal_action_bitmask)
        prior_logits, value = model.apply(
            {"params": params}, observation, legal
        )
        if config.get("exp_use_all_temp", False):
            tau = float(config["exp_all_temp"])
            prior_logits = jnp.where(
                legal, prior_logits / tau, jnp.finfo(prior_logits.dtype).min
            )
        rows = jnp.arange(batch_size)
        reward = env_state.rewards[rows, previous_player]
        discount = jnp.where(env_state.terminated, 0, -1).astype(jnp.float32)
        value = jnp.where(env_state.terminated, 0, value).astype(jnp.float32)
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=value,
            ),
            env_state,
        )

    def run(rng_key, env_state, params):
        _, search_key = jax.random.split(rng_key)
        root = root_fn(params, env_state)
        invalid_actions = ~_unpack_bitmask_vmap(env_state.legal_action_bitmask)
        terminal = env_state.terminated
        invalid_actions = jnp.where(
            terminal[:, None], jnp.ones_like(invalid_actions), invalid_actions
        )
        invalid_actions = invalid_actions.at[:, 0].set(
            jnp.where(terminal, False, invalid_actions[:, 0])
        )
        return mctx.gumbel_muzero_policy_opt(
            params=params,
            invalid_actions=invalid_actions,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_k_actions=int(config["mcts_num_k_actions"]),
            num_simulations=int(config["mcts_num_simulations"]),
            max_num_considered_actions=int(config["mcts_max_m"]),
            max_depth=int(config["game_max_steps"]),
            qtransform=qtransform,
            final_qtransform=final_qtransform,
            interior_qtransform=interior_qtransform,
            gumbel_scale=float(config["mcts_gumbel_scale"]),
            rehydrate_fields=bool(config["mcts_bnk_rehydrate_fields"]),
            return_search_tree=bool(config["mcts_return_search_tree"]),
            return_summary=bool(config["mcts_return_summary"]),
            use_opt_backward=bool(config["mcts_use_opt_backward"]),
            use_advantage_weights=bool(config["mcts_use_advantage_weights"]),
            advantage_scale=float(config["mcts_advantage_scale"]),
            use_puct_interior=bool(config["mcts_use_puct_interior"]),
        )

    return jax.jit(run), qtransform_config


class MctxPlayer:
    """A resident-batch KataModel player using MCTX's optimized JAX search."""

    def __init__(
        self,
        name: str,
        agent: dict[str, Any],
        batch_size: int,
        env,
        *,
        max_plies: int,
    ):
        self.name = name
        self.env = env
        devices = max(1, jax.device_count())
        if batch_size < devices or batch_size % devices:
            raise ValueError(
                f"MCTX batch_size must be divisible by {devices}; got {batch_size}"
            )
        checkpoint = Path(agent["checkpoint"]).expanduser().resolve()
        digest = sha256_file(checkpoint)
        cache_key = (str(checkpoint), digest)
        self.config = resolved_mctx_config(agent, env, max_plies)

        cached = _MODEL_CACHE.get(cache_key)
        if cached is None:
            params, model_config = load_checkpoint(str(checkpoint))
            self.config = apply_checkpoint_model_config(self.config, model_config)
            model, initialized = core.make_model(
                self.config,
                jax.random.PRNGKey(int(agent.get("seed", 42))),
                core.REPLICATED_SHARDING,
            )
            _validate_params(params, initialized["params"], checkpoint)
            params = jax.tree.map(
                lambda value: jax.device_put(value, core.REPLICATED_SHARDING),
                params,
            )
            cached = (params, model_config, model)
            _MODEL_CACHE[cache_key] = cached
        self.params, model_config, self.model = cached
        self.config = apply_checkpoint_model_config(self.config, model_config)
        self._run, self.qtransform_config = make_eval_mcts(
            self.config, env, self.model
        )
        self._seed = int(agent.get("seed", 42))
        self._calls = 0
        self._positions = 0
        self._seconds = 0.0
        self._warmup_seconds = 0.0
        self._compiled_sizes: set[int] = set()
        print(
            f"[{self.name}] MCTX opt: simulations="
            f"{self.config['mcts_num_simulations']} "
            f"max_m={self.config['mcts_max_m']} "
            f"k={self.config['mcts_num_k_actions']} "
            f"qtransform={self.qtransform_config}",
            flush=True,
        )

    def warmup(self, batch_size: int) -> None:
        started = time.perf_counter()
        state = self.env.init_dummy_estate(batch_size)
        state = jax.device_put(state, self.env.data_sharding)
        output = self._run(jax.random.PRNGKey(self._seed), state, self.params)
        jax.block_until_ready(output.action)
        self._warmup_seconds = time.perf_counter() - started
        self._compiled_sizes.add(batch_size)

    def play_actions(self, env_state, *, seed: int) -> np.ndarray:
        batch_size = int(jax.tree.leaves(env_state)[0].shape[0])
        started = time.perf_counter()
        output = self._run(
            jax.random.PRNGKey(np.uint32(seed)), env_state, self.params
        )
        actions = np.asarray(jax.device_get(output.action), dtype=np.int32)
        self._seconds += time.perf_counter() - started
        self._calls += 1
        self._positions += batch_size
        self._compiled_sizes.add(batch_size)
        return actions

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "mctx-opt",
            "batch_calls": self._calls,
            "positions": self._positions,
            "batch_wall_seconds": self._seconds,
            "warmup_seconds": self._warmup_seconds,
            "positions_per_second": (
                self._positions / self._seconds if self._seconds else 0.0
            ),
            "compiled_batch_sizes": sorted(self._compiled_sizes),
            "simulations": int(self.config["mcts_num_simulations"]),
            "qtransform": self.qtransform_config,
        }

    def close(self) -> None:
        pass
