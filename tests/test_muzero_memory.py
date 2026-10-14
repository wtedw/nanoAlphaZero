"""Buffer accounting stays global while memory queries stay host-local."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nanoalphazero import buffers


def test_storage_budget_only_queries_local_devices(monkeypatch):
    estimate = Mock(return_value={"replay": {"bytes_per_device": 100}})
    monkeypatch.setattr(buffers, "estimate_muzero_storage", estimate)
    monkeypatch.setattr(buffers.jax, "process_index", lambda: 1)
    remote = SimpleNamespace(process_index=0, memory_stats=Mock(side_effect=AssertionError("remote query")))
    local = SimpleNamespace(process_index=1, memory_stats=Mock(return_value={"bytes_limit": 1000}))
    buffers.check_muzero_storage_budget({}, [remote, local])
    estimate.assert_called_once_with({}, 2)
    local.memory_stats.assert_called_once()
    remote.memory_stats.assert_not_called()
    local.memory_stats.return_value = {"bytes_limit": 100}
    with pytest.raises(ValueError, match="persistent-memory budget"):
        buffers.check_muzero_storage_budget({}, [remote, local])
