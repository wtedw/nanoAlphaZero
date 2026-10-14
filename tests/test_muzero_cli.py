"""MuZero CLI parsing must not initialize JAX or change game defaults."""

import subprocess
import sys
from pathlib import Path

import pytest

from nanoalphazero.cli import MuZeroArgs, parse_muzero_args
from nanoalphazero.config import get_muzero_config


def test_typed_overrides_and_paths():
    args = parse_muzero_args([
        "--env", "hex6", "--platform", "cpu", "--num-iters", "300",
        "--replay-buffer-warmup-steps", "60", "--learning-rate", "0.002",
        "--ckpt.save", "artifacts/test.safetensors",
    ])
    assert isinstance(args, MuZeroArgs)
    assert args.num_iters == 300
    assert args.replay_buffer_warmup_steps == 60
    assert args.learning_rate == 0.002
    assert args.ckpt.save == Path("artifacts/test.safetensors")
    assert args.conv_width is None


@pytest.mark.parametrize("name", [
    "enable_sharding", "enable_wandb", "exp_bnk_action_weights",
])
def test_boolean_overrides_preserve_three_states(name):
    flag = name.replace("_", "-")
    assert getattr(parse_muzero_args([]), name) is None
    assert getattr(parse_muzero_args(["--" + flag]), name) is True
    assert getattr(parse_muzero_args(["--no-" + flag]), name) is False


@pytest.mark.parametrize("env", ["hex6", "chess"])
def test_omitted_options_inherit_game_defaults(env):
    args = parse_muzero_args(["--env", env])
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("env", "muzero", "ckpt", "acpl", "platform", "distributed", "ladder_period")
                 and not k.startswith("hex_eval_")}
    overrides.update(args.muzero.overrides())
    assert get_muzero_config(env, **overrides) == get_muzero_config(env)


@pytest.mark.parametrize("argv", [
    ["--env", "not-a-game"], ["--platform", "invalid"],
    ["--num-iters", "many"], ["--muzero-warmup-cycles", "2"],
])
def test_tyro_rejects_invalid_cli(argv):
    with pytest.raises(SystemExit):
        parse_muzero_args(argv)


def test_parser_does_not_import_jax():
    subprocess.run([
        sys.executable, "-c",
        "import sys; from nanoalphazero.cli import parse_muzero_args; "
        "parse_muzero_args(['--platform', 'cpu']); assert 'jax' not in sys.modules",
    ], check=True)


def test_muzero_config_group():
    args = parse_muzero_args(["--muzero.predict-value-only", "--muzero.unroll-steps", "7",
                             "--muzero.no-remat-blocks"])
    assert args.muzero.overrides() == {
        "muzero_predict_value_only": True, "muzero_unroll_steps": 7,
        "muzero_exploration_losses": "policy-transition",
        "muzero_remat_blocks": False,
        "muzero_train_on_terminal_states": False,
        "muzero_terminal_policy": "none",
    }
    with pytest.raises(SystemExit):
        parse_muzero_args(["--muzero-discount", "0.99"])


@pytest.mark.parametrize("env", ["hex4", "chess"])
def test_value_only_policy_transition_defaults(env):
    args = parse_muzero_args(["--env", env])
    config = get_muzero_config(env, **args.muzero.overrides())
    assert config["muzero_predict_value_only"] is True
    assert config["muzero_exploration_losses"] == "policy-transition"
    assert not config.get("exp_self_consistency", False)
    old = parse_muzero_args(["--muzero.no-predict-value-only",
                            "--muzero.exploration-losses", "none"])
    config = get_muzero_config(env, **old.muzero.overrides())
    assert config["muzero_predict_value_only"] is False
    assert config["muzero_exploration_losses"] == "none"


def test_existing_eval_output_is_preserved(tmp_path):
    with pytest.raises(ValueError, match="new directory"):
        parse_muzero_args([
            "--hex-eval-period", "50", "--hex-eval-engine-path", "/path/to/mohex",
            "--hex-eval-output", str(tmp_path),
        ])
