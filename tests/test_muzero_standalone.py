"""Verify the flattened export independently of package imports and Git."""

import ast
import importlib.util
import json
from pathlib import Path
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nanoalphazero.research.muzero import checkpoint
from nanoalphazero.research.muzero.model import build_model
from nanoalphazero.research.muzero.learning import loss
from nanoalphazero.research.muzero.search import latent_search


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def flat():
    spec = importlib.util.spec_from_file_location("flat_muzero_test", ROOT / "muzero.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_tree_equal(left, right):
    # Dataclass types intentionally differ between the two source locations.
    a, b = jax.tree.leaves(left), jax.tree.leaves(right)
    assert len(a) == len(b)
    for actual, expected in zip(a, b, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_no_package_imports_or_embedded_module_loader():
    tree = ast.parse((ROOT / "muzero.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("nanoalphazero")
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("nanoalphazero") for alias in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("exec", "eval", "__import__") or (
                node.func.id == "__import__" and node.args[0].value == "sys")


@pytest.mark.parametrize("env", [f"hex{size}" for size in range(4, 10)])
def test_training_defaults_need_no_config_file(flat, env):
    args = flat.standalone_train_parser().parse_args(["--env", env])
    raw = flat.standalone_train_settings(args)
    config = flat.resolve(raw)
    base = flat.CONFIG_FACTORIES[env]()
    assert config["platform"] == "tpu" and config["devices"] == 4
    assert config["network"] == "spatial" and config["data_pipeline"] == "staged"
    assert config["learning_rate"] == base["learning_rate"]
    assert config["train_batch_size"] == (4096 if env == "hex5" else min(4096, base["train_batch_size"]))
    assert config["selfplay_batch_size"] <= 4096 and config["heldout_batch_size"] <= 4096
    assert config["width"] <= 256 and config["depth"] <= 10
    assert config["roots"] <= 8 and config["survivors"] <= 4
    assert not config["initial_evaluation"]
    assert config["wandb"] and not config["save_checkpoints"]
    assert config["hex_eval_period"] == 0
    assert args.output is None
    assert flat.standalone_output(config) != flat.standalone_output(config)


def test_explicit_cli_overrides_respect_new_limits(flat):
    args = flat.standalone_train_parser().parse_args([
        "--env", "hex7", "--width", "512", "--depth", "32", "--roots", "16", "--survivors", "8",
        "--train-batch-size", "8192", "--selfplay-batch-size", "8192", "--heldout-batch-size", "8192",
        "--initial-evaluation", "--opening-coverage-streak", "3", "--hex-eval-period", "50",
        "--opening-value-mse-threshold", "0.01"])
    config = flat.resolve(flat.standalone_train_settings(args))
    assert (config["width"], config["depth"], config["roots"], config["survivors"]) == (256, 10, 8, 4)
    assert all(config[key] == 4096 for key in ("train_batch_size", "selfplay_batch_size", "heldout_batch_size"))
    assert config["initial_evaluation"] and config["opening_coverage_streak"] == 3
    assert config["opening_value_mse_threshold"] == .01


def test_cli_overrides_optional_toml_and_preset(flat, tmp_path):
    path = tmp_path / "overrides.toml"
    path.write_text('seed = 3\nlearning_rate = 0.002\n')
    args = flat.standalone_train_parser().parse_args([
        str(path), "--preset", "smoke-cpu", "--seed", "7", "--cycles", "3", "--no-wandb"])
    config = flat.resolve(flat.standalone_train_settings(args))
    assert config["seed"] == 7 and config["cycles"] == 3
    assert config["learning_rate"] == .002 and not config["wandb"]


@pytest.mark.parametrize("network", ["vector", "spatial"])
def test_model_search_loss_and_gradients_match_package(flat, network):
    config = flat.resolve({**flat.STANDALONE_PRESETS["smoke-cpu"], "network": network})
    config.update(num_actions=9, obs_shape=[3, 3, 2])
    original, exported = build_model(config), flat.build_model(config)
    observation = jnp.arange(36, dtype=jnp.float32).reshape(2, 3, 3, 2) % 2
    action = jnp.array([1, 2])
    params = original.init(jax.random.PRNGKey(12), observation, action)["params"]
    exported_params = exported.init(jax.random.PRNGKey(12), observation, action)["params"]
    assert_tree_equal(params, exported_params)
    assert_tree_equal(original.apply({"params": params}, observation, action),
                      exported.apply({"params": params}, observation, action))
    episodes = dict(observation=jnp.tile(observation[:, None], (1, 3, 1, 1, 1)),
                    action=jnp.array([[0, 1, 2], [2, 1, 0]]),
                    reward=jnp.array([[0., 0., 1.], [0., 0., -1.]]),
                    policy=jnp.ones((2, 3, 9)) / 9, value=jnp.array([[1., -1., 1.], [-1., 1., -1.]]),
                    length=jnp.array([3, 2]), terminal=jnp.array([True, False]), bootstrap=jnp.array([0., .3]))
    batch = flat.sequences(episodes, jax.random.PRNGKey(1), 3, starts=jnp.array([1, 1]))
    original_result = jax.value_and_grad(loss, argnums=1, has_aux=True)(original, params, batch)
    exported_result = jax.value_and_grad(flat.loss, argnums=1, has_aux=True)(exported, params, batch)
    assert_tree_equal(original_result, exported_result)
    legal = jnp.ones((2, 9), dtype=bool).at[:, 0].set(False)
    args = (params, observation, legal, jax.random.PRNGKey(9))
    assert_tree_equal(latent_search(original, *args, roots=4, survivors=2),
                      flat.latent_search(exported, *args, roots=4, survivors=2))


@pytest.mark.parametrize("network,env", [("vector", "ttt"), ("spatial", "hex4")])
def test_copied_script_trains_and_checkpoint_loads_without_repo(tmp_path, flat, network, env):
    copied = tmp_path / "muzero.py"
    copied.write_bytes((ROOT / "muzero.py").read_bytes())
    extra = ["--remat-blocks", "--remat-unroll"] if network == "spatial" else []
    # Block even accidental lazy package imports; this also runs outside Git.
    runner = '''import runpy, sys
class BlockPackage:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "nanoalphazero" or fullname.startswith("nanoalphazero."):
            raise AssertionError("Standalone script imported " + fullname)
sys.meta_path.insert(0, BlockPackage())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
    result = subprocess.run([sys.executable, "-c", runner, str(copied), "train",
                             "--preset", "smoke-cpu", "--save", "--network", network,
                             "--env", env, "--eval-period", "0", *extra],
                            cwd=tmp_path, env={**os.environ, "JAX_PLATFORMS": "cpu",
                            "XLA_FLAGS": "--xla_force_host_platform_device_count=4"},
                            text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = list((tmp_path / "artifacts").glob("muzero-*"))
    assert len(outputs) == 1
    output = outputs[0]
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["runner_state/n_updates"] == 4
    assert metrics[-1]["muzero/drain/inserted_positions"] > 0
    assert all(np.isfinite(record["total_loss"]) for record in metrics)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["devices"]) == 4
    saved = output / "cycle-000002.safetensors"
    package_params, package_config = checkpoint.load_params(saved)
    flat_params, flat_config = flat.load_params(saved)
    assert package_config == flat_config
    assert_tree_equal(package_params, flat_params)
