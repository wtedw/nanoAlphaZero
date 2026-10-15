"""Startup validation without initializing an accelerator backend."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nanoalphazero.distributed import DistributedArgs, initialize_distributed


@pytest.fixture
def runtime(monkeypatch):
    distributed = SimpleNamespace(is_initialized=Mock(return_value=False), initialize=Mock())
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(distributed=distributed))
    return distributed


def test_disabled_does_not_touch_jax(runtime):
    initialize_distributed(DistributedArgs())
    runtime.is_initialized.assert_not_called()
    runtime.initialize.assert_not_called()


def test_pod_discovery(runtime):
    initialize_distributed(DistributedArgs(enabled=True))
    runtime.initialize.assert_called_once_with(initialization_timeout=120)


def test_explicit_topology(runtime):
    initialize_distributed(DistributedArgs(True, "localhost:1234", 2, 1, 30))
    runtime.initialize.assert_called_once_with(
        coordinator_address="localhost:1234", num_processes=2,
        process_id=1, initialization_timeout=30,
    )


@pytest.mark.parametrize("args", [
    DistributedArgs(True, "localhost:1234"),
    DistributedArgs(True, "localhost:1234", 2, 2),
    DistributedArgs(True, "localhost:1234", 0, 0),
    DistributedArgs(False, "localhost:1234", 2, 0),
    DistributedArgs(initialization_timeout=0),
])
def test_invalid_topology_fails_before_jax(args, runtime):
    with pytest.raises(ValueError):
        initialize_distributed(args)
    runtime.initialize.assert_not_called()


def test_repeated_initialization_fails(runtime):
    runtime.is_initialized.return_value = True
    with pytest.raises(RuntimeError, match="already initialized"):
        initialize_distributed(DistributedArgs(enabled=True))
    runtime.initialize.assert_not_called()


def test_cli_distributed_options():
    from nanoalphazero.cli import parse_muzero_args

    args = parse_muzero_args([
        "--distributed.enabled", "--distributed.coordinator-address", "localhost:1234",
        "--distributed.num-processes", "2", "--distributed.process-id", "0",
    ])
    assert args.distributed == DistributedArgs(True, "localhost:1234", 2, 0)


def test_multihost_unsupported_features_fail_explicitly():
    from nanoalphazero.distributed import validate_muzero_run

    config = {"enable_sharding": True, "eval_period": 0}
    validate_muzero_run(config, 4)
    validate_muzero_run({"enable_trackio": True}, 1, checkpoint=True)
    with pytest.raises(ValueError, match="global sharding"):
        validate_muzero_run({}, 4)
    validate_muzero_run({**config, "enable_trackio": True}, 4)
    validate_muzero_run({**config, "enable_wandb": True}, 4)
    validate_muzero_run({**config, "acpl": {"period": 10}}, 4)
    for option in ({"eval_period": 10},):
        with pytest.raises(ValueError, match="disabled"):
            validate_muzero_run({**config, **option}, 4)
    validate_muzero_run(config, 4, checkpoint=True)
    for options in ({"hex_evaluation": True},):
        with pytest.raises(ValueError, match="disabled"):
            validate_muzero_run(config, 4, **options)
