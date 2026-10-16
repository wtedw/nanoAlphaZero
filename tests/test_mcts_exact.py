"""Exact search, root diagnostics, and post-search BNK targets.

Reference tests require the pinned upstream checkout renamed to mctx_ref on
PYTHONPATH (see docs/muzero-components.md); never use the installed mctx fork.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero.mcts import RootFnOutput, RecurrentFnOutput, compress_muzero_policy
from nanoalphazero.mcts_exact import gumbel_muzero_policy_node, search, QT


def problem(seed=0, actions=20):
    keys = jax.random.split(jax.random.PRNGKey(seed), 7)
    params = dict(g=jax.random.normal(keys[0], (8 + actions, 8)) / 3,
                  p=jax.random.normal(keys[1], (8, actions)),
                  v=jax.random.normal(keys[2], (8,)))
    root = RootFnOutput(prior_logits=jax.random.normal(keys[3], (8, actions)),
                        value=jnp.tanh(jax.random.normal(keys[4], (8,))),
                        embedding=jax.random.normal(keys[5], (8, 8)))
    invalid = jnp.arange(actions)[None] >= jnp.array([0, 1, 2, 3, 7, 8, 16, actions])[:, None]

    def recurrent(params, key, action, embedding):
        h = jnp.tanh(jnp.concatenate([embedding, jax.nn.one_hot(action, actions)], -1) @ params['g'])
        value = jnp.tanh(h @ params['v'])
        return RecurrentFnOutput(reward=value * .1, discount=jnp.full_like(value, -1),
                                 prior_logits=h @ params['p'], value=value), h
    return params, keys[6], root, recurrent, invalid


@pytest.mark.parametrize('budget,roots', [(12, 8), (32, 16), (7, 16), (64, 16)])
@pytest.mark.parametrize('rescale', [False, True])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_upstream_tree_and_policy_equivalence(budget, roots, rescale, seed):
    ref = pytest.importorskip('mctx_ref')
    from mctx_ref._src.qtransforms import qtransform_completed_by_mix_value

    params, key, root, recurrent, invalid = problem(seed)
    kw = dict(max_num_considered_actions=roots, invalid_actions=invalid,
              qtransform=functools.partial(qtransform_completed_by_mix_value,
                  value_scale=.1 if seed == 1 else 1., rescale_values=rescale))
    expected = jax.jit(lambda k: ref.gumbel_muzero_policy(params, k, root, recurrent, budget, **kw))(key)
    actual = jax.jit(lambda k: gumbel_muzero_policy_node(params, k, root, recurrent, budget, **kw))(key)
    np.testing.assert_array_equal(actual.action, expected.action)
    # Different fusion of the final softmax can change tiny probabilities even
    # on CPU. Tree state and decisions must remain bit-identical.
    np.testing.assert_allclose(actual.action_weights, expected.action_weights, atol=2e-7, rtol=2e-6)
    for a, b in zip(jax.tree.leaves(actual.search_tree), jax.tree.leaves(expected.search_tree), strict=True):
        np.testing.assert_array_equal(a, b)


def test_bnk_and_diagnostics_preserve_search():
    params, key, root, recurrent, invalid = problem()
    def run(diagnostics):
        return jax.jit(lambda k: search(params, k, root, recurrent,
            num_simulations=32, num_root_considered=16, invalid_actions=invalid,
            gumbel_scale=1., return_diagnostics=diagnostics))(key)
    output, diagnostic = run(False), run(True)
    for field in ['action', 'action_weights', 'visit_counts']:
        np.testing.assert_array_equal(getattr(output, field), getattr(diagnostic, field))
    np.testing.assert_array_equal(output.visit_counts.sum(-1), 32)
    assert np.isfinite(output.action_weights).all()
    np.testing.assert_allclose(output.action_weights.sum(-1), 1., atol=2e-7)
    assert not np.asarray(invalid)[np.arange(1, 8), np.asarray(output.action)[1:]].any()
    for k in [1, 8, 20]:
        packed, mass = compress_muzero_policy(output, k)
        np.testing.assert_array_equal(packed.action, output.action)
        np.testing.assert_array_equal(packed.visit_counts, output.visit_counts)
        w, ids = jax.lax.top_k(output.action_weights, k)
        np.testing.assert_array_equal(packed.bnk_k_indices, ids)
        np.testing.assert_array_equal(packed.bnk_action_weights, w / w.sum(-1, keepdims=True))
        np.testing.assert_array_equal(mass, w.sum(-1))
    np.testing.assert_allclose(jax.nn.softmax(diagnostic.diagnostics['search_logit']),
                               output.action_weights, atol=2e-7)


def test_factory_dispatches_exact_and_honors_budget():
    from nanoalphazero.config import get_muzero_config
    from nanoalphazero.core import make_env
    from nanoalphazero.model import make_muzero_model
    from nanoalphazero.mcts import make_muzero_mcts

    config = get_muzero_config('ttt', enable_sharding=False, conv_width=4,
        conv_depth=1, mcts_variant='gumbel', mcts_num_simulations=8,
        mcts_num_root_considered=4, mcts_num_k_actions=4, exp_bnk_action_weights=True)
    env = make_env(config)
    config.update(game_obs_shape=list(env.obs_shape), game_num_actions=env.num_actions)
    model, variables = make_muzero_model(config, jax.random.PRNGKey(0))
    state = env.init(jax.random.split(jax.random.PRNGKey(1), 4))
    fn = make_muzero_mcts(config, env, model)
    for budget in [None, 12]:
        out = fn(jax.random.PRNGKey(2), state, variables['params'], 1., 4, budget)
        np.testing.assert_array_equal(out.visit_counts.sum(-1), budget or 8)
        assert out.bnk_k_indices.shape == (4, 4)
