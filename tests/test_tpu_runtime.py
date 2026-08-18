import jax
import jax.numpy as jnp

from nanoalphazero import config, core
from nanoalphazero.eval.chess.action import PgxAction
from nanoalphazero.eval.chess.mctx_player import (
    build_qtransforms,
    make_eval_mcts,
    make_pgx1_env,
)


def test_chess_model_and_environment_match_main_shapes():
    env_config = config.get_chess_config()
    env_config.update(
        katago_preset="b1c8nbt",
        conv_depth=1,
        conv_width=8,
        enable_sharding=False,
    )
    env = core.make_env(env_config)
    env_config["game_obs_shape"] = env.obs_shape
    env_config["game_num_actions"] = env.num_actions
    model, variables = core.make_model(env_config, jax.random.PRNGKey(0), None)

    assert env.obs_shape == (8, 8, 119)
    assert env.num_actions == 4672
    assert model.c_p1 == core.resolve_preset("b1c8nbt")["c_p1"]
    assert variables["params"]["ChessPolicyHead_0"]["Conv_2"]["kernel"].shape[-1] == 73


def test_mctx_opt_returns_legal_actions_on_four_device_resident_batch():
    env = make_pgx1_env()
    batch_size = jax.device_count()
    assert batch_size == 4
    model_config = config.get_chess_config()
    model_config.update(
        katago_preset="b1c8nbt",
        conv_depth=1,
        conv_width=8,
        game_obs_shape=env.obs_shape,
        game_num_actions=env.num_actions,
        game_max_steps=16,
        enable_sharding=True,
        mcts_variant="opt",
        mcts_num_simulations=4,
        mcts_max_m=4,
        mcts_num_k_actions=4,
        mcts_visit_exponent=1.0,
        mcts_visit_aggregator="max",
        mcts_maxvisit_init=50.0,
        mcts_value_scale=1.0,
        mcts_rescale_values=False,
        mcts_use_mixed_value=True,
        mcts_gumbel_scale=0.0,
        mcts_use_opt_backward=True,
        mcts_bnk_rehydrate_fields=False,
        mcts_return_search_tree=False,
        mcts_return_summary=False,
        mcts_use_advantage_weights=False,
        mcts_advantage_scale=1.0,
        mcts_use_puct_interior=False,
    )
    model, variables = core.make_model(
        model_config, jax.random.PRNGKey(0), core.REPLICATED_SHARDING
    )
    run, qtransform = make_eval_mcts(model_config, env, model)
    state = jax.device_put(
        env.init_dummy_estate(batch_size), core.DATA_PARALLEL_SHARDING
    )
    output = run(jax.random.PRNGKey(1), state, variables["params"])
    actions = jax.device_get(output.action)
    board = __import__("chess").Board()

    assert actions.shape == (batch_size,)
    assert all(PgxAction.decode(int(action), board) in board.legal_moves for action in actions)
    assert qtransform["root"]["visit_aggregator"] == "max"


def test_interior_zero_exponent_is_not_replaced_by_root_value():
    settings = {
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_maxvisit_init": 50.0,
        "mcts_use_mixed_value": True,
        "mcts_visit_exponent": 1.0,
        "mcts_visit_aggregator": "max",
        "mcts_visit_exponent_interior": 0.0,
        "mcts_visit_aggregator_interior": "sum",
    }
    _, _, _, resolved = build_qtransforms(settings)
    assert resolved["interior"]["visit_exponent"] == 0.0
    assert resolved["interior"]["visit_aggregator"] == "sum"
