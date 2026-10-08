"""Checkpoint CLI grouping without initializing training or writing snapshots."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nanoalphazero.cli import parse_muzero_args
from nanoalphazero.config import get_muzero_config


@pytest.mark.parametrize(
    "env,options,expected_mode,expected_period",
    [
        ("chess", [], False, None),
        ("hex4", [], True, None),
        ("hex4", ["--ckpt.mode", "compact", "--ckpt.every", "400", "--ladder-period", "0"], False, 400),
        ("chess", ["--ckpt.mode", "full", "--ckpt.every", "0"], True, 0),
    ],
)
def test_checkpoint_options_reach_runner(monkeypatch, env, options, expected_mode, expected_period):
    path = Path(__file__).resolve().parents[1] / "muzero.py"
    spec = importlib.util.spec_from_file_location("muzero_checkpoint_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    def assemble(config, rng):
        captured["config"] = config
        return "algorithm"

    def run(algorithm, **kwargs):
        assert algorithm == "algorithm"
        captured.update(kwargs)

    monkeypatch.setattr(module, "make_muzero", assemble)
    monkeypatch.setattr(module, "run_muzero", run)
    module.main([
        "--env", env, "--platform", "cpu",
        "--ckpt.save", "artifacts/output.safetensors",
        "--ckpt.resume", "artifacts/input.safetensors", *options,
    ])
    config = captured["config"]
    assert config["muzero_checkpoint_replay"] is expected_mode
    assert config["ckpt_period"] == (
        get_muzero_config(env)["ckpt_period"] if expected_period is None else expected_period
    )
    assert "ckpt" not in config
    assert "ladder_period" not in config
    assert config["eval_period"] == (0 if "--ladder-period" in options else get_muzero_config(env)["eval_period"])
    assert captured["checkpoint_path"] == Path("artifacts/output.safetensors")
    assert captured["resume"] == Path("artifacts/input.safetensors")


def test_resume_does_not_enable_saving():
    args = parse_muzero_args(["--ckpt.resume", "artifacts/input.safetensors"])
    assert args.ckpt.save is None
    assert args.ckpt.every is None
    assert args.ckpt.mode is None


def test_checkpoint_help_without_jax():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
import muzero
try:
    muzero.main(["--help"])
except SystemExit as error:
    assert error.code == 0
assert "jax" not in sys.modules
"""],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, check=True,
    )
    assert "ckpt options" in result.stdout
    for name in ("save", "resume", "every", "mode"):
        assert f"--ckpt.{name}" in result.stdout
