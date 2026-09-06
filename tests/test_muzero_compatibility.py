"""Full rollout-cap smoke gates for every environment; not strength tests."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from nanoalphazero.config import CONFIG_FACTORIES
from nanoalphazero.research.muzero import checkpoint
from nanoalphazero.research.muzero.learning import make_collect, train_step
from nanoalphazero.research.muzero.model import MuZero
from nanoalphazero.research.muzero.replay import sequences
from nanoalphazero.research.muzero.search import make_search, legal_actions
from nanoalphazero.research.muzero.staging import make_staging


@pytest.mark.parametrize("name,network", [(name, "vector") for name in CONFIG_FACTORIES] +
                         [(name, "spatial") for name in ("hex7", "hex8", "hex9")])
def test_environment_cycle(name, network, tmp_path):
    from nanoalphazero.core import make_env
    base = CONFIG_FACTORIES[name]()
    env = make_env(base)
    config = dict(selfplay_batch_size=4, roots=min(4, env.num_actions), survivors=2,
                  discount=1., exploration_moves=2, max_steps=base["game_max_steps"])
    if network == "spatial":
        from nanoalphazero.research.muzero.spatial import SpatialMuZero
        model = SpatialMuZero(env.num_actions, base["env_id"], 8, 1, "mish", True)
    else:
        model = MuZero(env.num_actions, 8, 0)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, *env.obs_shape)), jnp.zeros(1, jnp.int32))["params"]
    search = make_search(model, env, config)
    collect = jax.jit(make_collect(env, model, search, config))
    episodes = collect(params, jax.random.PRNGKey(1))
    assert not bool(jnp.any(episodes["illegal"]))
    assert bool(jnp.all(jnp.isfinite(episodes["value"])))
    staging_config = {**config, "staging_batches": 2, "consume_size": 4,
                      "train_batch_size": 4096 if name == "hex5" else 4,
                      "replay_positions": 128, "unroll": 5}
    init, add, consume, drain, replay = make_staging(staging_config)
    staged = init(jax.tree.map(lambda x: x[0], episodes))
    staged, data_metrics = add(staged, episodes)
    assert not data_metrics["staging/malformed_episodes"]
    dummy = jax.eval_shape(consume, staged, jax.random.PRNGKey(2))[1]
    replay_state = replay.init(jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), dummy))
    staged, replay_state, _, data_metrics = jax.jit(drain)(staged, replay_state, jax.random.PRNGKey(2))
    assert data_metrics["drain/inserted_positions"] > 0
    batch = jax.tree.map(lambda x: x[:, 0], replay.sample(replay_state, jax.random.PRNGKey(5)).experience)
    assert bool(jnp.all(batch["policy_mask"][:, 0]))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    state, metrics = jax.jit(lambda s, b: train_step(model, s, b))(state, batch)
    assert all(np.isfinite(float(v)) for v in metrics.values())
    path = tmp_path / f"{name}.safetensors"
    checkpoint_tree = {"params": state.params, "staging": staged, "replay": vars(replay_state)}
    checkpoint.save(path, checkpoint_tree, dict(env=name, obs_shape=list(env.obs_shape)))
    loaded, _ = checkpoint.load(path, checkpoint_tree)
    for old, new in zip(jax.tree.leaves(checkpoint_tree), jax.tree.leaves(loaded)):
        np.testing.assert_array_equal(old, new)
    root = env.init_dummy_estate(4)
    output = search(jax.random.PRNGKey(3), root, loaded["params"], 0., 4)
    assert bool(jnp.all(legal_actions(root)[jnp.arange(4), output.action]))
    # Verify observed next-player semantics independently of latent search.
    following = env.step(root, output.action)
    assert bool(jnp.all(following.current_player != root.current_player))
    assert following.rewards.shape == (4, 2)
    from nanoalphazero.training import run_eval_match
    openings = np.flatnonzero(np.asarray(legal_actions(root)[0]))[:4]
    evaluation_config = {**base, "enable_sharding": False}
    w, d, l = run_eval_match(search, env, evaluation_config, loaded["params"], None,
                             opening_actions=openings, key=jax.random.PRNGKey(4),
                             max_plies=base["game_max_steps"])
    assert w + d + l == len(openings)
    jax.clear_caches()
