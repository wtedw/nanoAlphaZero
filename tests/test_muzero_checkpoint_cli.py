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
        ("chess", [], "full", None),
        ("hex4", [], "full", None),
        ("hex4", ["--ckpt.every", "400", "--ladder-period", "0"], "full", 400),
        ("chess", ["--ckpt.mode", "partial", "--ckpt.every", "0"], "partial", 0),
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
    import nanoalphazero.ckpt as ckpt
    monkeypatch.setattr(ckpt, "inspect_checkpoint", lambda path: {"mode": "partial"})
    monkeypatch.setattr(ckpt, "validate_resume_config",
                        lambda path, config, mode: captured.update(resume_mode=mode))
    module.main([
        "--env", env, "--platform", "cpu",
        "--ckpt.save", "artifacts/output.ckpt",
        "--ckpt.resume", "artifacts/input.safetensors", *options,
    ])
    config = captured["config"]
    assert "muzero_checkpoint_replay" not in config
    assert captured["checkpoint_mode"] == expected_mode
    assert captured["resume_mode"] == "partial"
    assert config["ckpt_period"] == (
        get_muzero_config(env)["ckpt_period"] if expected_period is None else expected_period
    )
    assert "ckpt" not in config
    assert "ladder_period" not in config
    assert config["eval_period"] == (0 if "--ladder-period" in options else get_muzero_config(env)["eval_period"])
    assert captured["checkpoint_path"] == Path("artifacts/output.ckpt")
    assert captured["resume"] == Path("artifacts/input.safetensors")
    assert not hasattr(parse_muzero_args([]).ckpt, "compression")


def test_resume_does_not_enable_saving():
    args = parse_muzero_args(["--ckpt.resume", "artifacts/input.safetensors"])
    assert args.ckpt.save is None
    assert args.ckpt.every is None
    assert args.ckpt.mode == "full"


def test_value_only_resume_semantics(monkeypatch):
    import nanoalphazero.ckpt.inspection as inspection
    config = get_muzero_config("hex4")
    old = {k: v for k, v in config.items() if k != "muzero_predict_value_only"}
    monkeypatch.setattr(inspection, "inspect_checkpoint",
                        lambda _: dict(mode="full", config=old))
    inspection.validate_resume_config("unused", config)
    with pytest.raises(ValueError, match="identical resolved configuration"):
        inspection.validate_resume_config("unused", {**config, "muzero_predict_value_only": True})


def test_value_only_requires_undiscounted_outcomes():
    from nanoalphazero.config import validate_muzero_config
    config = get_muzero_config("hex4", muzero_predict_value_only=True, muzero_discount=0.9)
    with pytest.raises(ValueError, match="undiscounted"):
        validate_muzero_config(config, 16)


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
    assert "--ckpt.compression" not in result.stdout
    for name in ("save", "resume", "every", "mode"):
        assert f"--ckpt.{name}" in result.stdout
