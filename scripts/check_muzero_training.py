"""Numerical comparison of checkpoint strategies on the same MuZero batch/state."""

import json
import time


def check_training_equivalence(module, config, candidate_model, model_state, batch, output):
    import jax
    import numpy as np
    import optax

    reference_model = candidate_model.clone(remat_blocks=True)

    def evaluate(model, remat):
        def loss(params, batch):
            return module.muzero_loss(
                model, params, batch, remat=remat,
                self_consistency_weight=(config.get("self_consistency_weight", 1.0)
                                         if config.get("exp_self_consistency", False) else 0.0),
                value_only=config.get("muzero_predict_value_only", False))

        @jax.jit
        def fn(params, opt_state, batch):
            (loss_value, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params, batch)
            updates, next_opt = model_state.tx.update(grads, opt_state, params)
            return loss_value, metrics, grads, updates, optax.apply_updates(params, updates), next_opt

        return fn

    def compare(actual, expected, *, atol, rtol):
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        actual, expected = jax.device_get((actual, expected))
        maximum, square_error, square_reference, mismatches, count = 0.0, 0.0, 0.0, 0, 0
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            x, y = np.asarray(x), np.asarray(y)
            assert x.shape == y.shape and x.dtype == y.dtype
            assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
            delta = x.astype(np.float64) - y.astype(np.float64)
            maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
            square_error += float(np.sum(delta * delta))
            square_reference += float(np.sum(y.astype(np.float64) ** 2))
            mismatches += int(np.count_nonzero(np.abs(delta) > atol + rtol * np.abs(y)))
            count += x.size
        return dict(max_abs=maximum, relative_l2=(square_error / max(square_reference, 1e-30)) ** .5,
                    mismatches=mismatches, elements=count, atol=atol, rtol=rtol)

    params, opt_state = model_state.params, model_state.opt_state
    print("Checking loss, gradients, AdamW updates and optimizer state against double-checkpoint reference", flush=True)
    started = time.perf_counter()
    reference_fn = evaluate(reference_model, True)
    reference = jax.block_until_ready(reference_fn(params, opt_state, batch))
    print(f"Reference numerics computed in {time.perf_counter() - started:.2f}s", flush=True)
    candidate_fn = evaluate(candidate_model, config.get("muzero_remat_unroll", False))
    candidate = jax.block_until_ready(candidate_fn(params, opt_state, batch))
    report = {}
    # Checkpoint scheduling changes floating-point accumulation order. Absolute
    # tolerances protect near-zero entries; whole-tree norm gates below prevent
    # them from hiding substantial aggregate gradient or optimizer-update drift.
    for name, x, y, atol, rtol in zip(
        ("loss", "metrics", "gradients", "updates", "parameters", "optimizer_state"),
        candidate, reference,
        (1e-5, 1e-5, 1e-5, 1e-8, 1e-6, 1e-6),
        (1e-5, 1e-5, 1e-3, 1e-3, 1e-5, 1e-3), strict=True,
    ):
        report[name] = compare(x, y, atol=atol, rtol=rtol)
    report["passed"] = (all(row["mismatches"] == 0 for row in report.values())
                        and report["gradients"]["relative_l2"] < 1e-3
                        and report["updates"]["relative_l2"] < 1e-4)
    (output / "equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    assert report["passed"], "Candidate exceeds numerical tolerances; inspect equivalence.json"
    del candidate, reference
    reference_fn.clear_cache()
    candidate_fn.clear_cache()
