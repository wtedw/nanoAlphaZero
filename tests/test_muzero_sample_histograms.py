"""Optimizer provenance is logged per root, not per unrolled position."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from nanoalphazero.muzero_metrics import training_batch_arrays, training_batch_histograms


def test_sample_arrays_and_masked_cycle_histograms():
    batch = SimpleNamespace(
        row_id=jnp.array([1, 2, 3], dtype=jnp.uint32),
        col_id=jnp.array([12, 13, 14], dtype=jnp.uint32),
        game_id=jnp.array([101, 102, 103], dtype=jnp.uint32),
        global_step_id=jnp.array([92, 93, 94], dtype=jnp.uint32),
        ep_step=jnp.array([4, 5, 6]),
        # Exploratory root with a supervised future is still a valid window.
        policy_mask=jnp.array([[False, True], [False, False], [True, True]]),
    )
    arrays = training_batch_arrays(batch, jnp.int32(100))
    np.testing.assert_array_equal(arrays["age"], [8, 7, 6])
    # Two optimizer batches: preserve repetitions, never average identifiers.
    stacked = {name: np.stack([value, value]) for name, value in arrays.items()}
    histograms = training_batch_histograms(stacked)
    np.testing.assert_array_equal(histograms["train_batch/sampled_row_ids_distribution"], [1, 3, 1, 3])
    np.testing.assert_array_equal(histograms["train_batch/sampled_col_ids_distribution"], [12, 14, 12, 14])
    np.testing.assert_array_equal(histograms["train_batch/sampled_game_ids_distribution"], [101, 103, 101, 103])
    np.testing.assert_array_equal(histograms["train_batch/ep_step_distribution"], [4, 6, 4, 6])
    np.testing.assert_array_equal(histograms["train_batch/sample_age_distribution"], [8, 6, 8, 6])
    stacked["valid"][:] = False
    assert training_batch_histograms(stacked) == {}
