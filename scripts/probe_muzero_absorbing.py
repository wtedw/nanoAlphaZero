"""Audit absorbing supervision from a compressed chess replay on CPU."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import zstandard as zstd
from nanoalphazero.ckpt import load_model
from nanoalphazero.ckpt.compressed import _lock, _generation
from nanoalphazero.buffers import combine_observation_vmap


def replay_sample(path):
    """Inspect masks globally; read only selected board/action chunks."""
    rng = np.random.default_rng(42)
    with _lock(path, exclusive=False):
        directory = _generation(path)
        manifest = json.loads((directory / "manifest.json").read_text())
        prefix = "['runner_state'].replay_buffer_state.experience."
        fields = {f["path"].removeprefix(prefix): f for f in manifest["fields"] if f["path"].startswith(prefix)}
        def chunks(name):
            field = fields[name]
            for shard in field["shards"]:
                offset = np.array([idx[0] for idx in shard["index"]])
                for chunk in shard["chunks"]:
                    yield offset + chunk["start"], chunk, field["dtype"]
        def read(chunk, dtype):
            raw = zstd.ZstdDecompressor().decompress((directory / chunk["file"]).read_bytes())
            return np.frombuffer(raw, dtype=dtype).reshape(chunk["shape"])
        def full(name):
            out = np.empty(fields[name]["shape"], dtype=fields[name]["dtype"])
            for start, chunk, dtype in chunks(name):
                out[tuple(slice(s, s+n) for s, n in zip(start, chunk["shape"]))] = read(chunk, dtype)
            return out
        policy, value = full("policy_mask"), full("value_mask")
        # In this checkpoint explore positions mask both heads. Consequently
        # value-valid/policy-invalid positions identify absorbing padding.
        padding = value & ~policy
        has_padding = padding.any(-1)
        valid = value.any(-1)
        stats = dict(valid_windows=int(valid.sum()), padded_windows=int(has_padding.sum()),
                     padding_by_depth=padding.sum((0, 1)).tolist(),
                     valid_values_by_depth=value.sum((0, 1)).tolist(),
                     padding_fraction=float(padding.sum()/value.sum()))
        coordinates = []
        for pool in (has_padding, valid & ~has_padding):
            flat = np.flatnonzero(pool)
            chosen = rng.choice(flat, 128, replace=False)
            coordinates.extend(zip(*np.unravel_index(chosen, pool.shape)))
        coordinates = np.array(coordinates)
        samples = {}
        for name in ("board_bool", "board_float", "action", "reward", "transition_reward", "policy_mask", "value_mask", "reward_mask"):
            samples[name] = np.empty((len(coordinates), fields[name]["shape"][-1]), dtype=fields[name]["dtype"])
            for start, chunk, dtype in chunks(name):
                end = start + chunk["shape"]
                selected = np.flatnonzero(((coordinates >= start[:2]) & (coordinates < end[:2])).all(-1))
                if len(selected):
                    data = read(chunk, dtype)
                    local = coordinates[selected] - start[:2]
                    samples[name][selected] = data[local[:, 0], local[:, 1]]
        samples["coordinates"] = coordinates
        return directory.name, stats, samples


def describe(x):
    x = np.asarray(x)
    return dict(n=x.size, mean_abs=float(np.abs(x).mean()), rms=float(np.sqrt(np.mean(x*x))),
                p95_abs=float(np.quantile(np.abs(x), .95)), max_abs=float(np.abs(x).max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-from", type=Path, help="Reuse a fixed sample/report directory across model checkpoints")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.sample_from:
        reference = json.loads((args.sample_from / "report.json").read_text())
        stats = reference["replay"]
        sample = dict(np.load(args.sample_from / "sample.npz"))
        # NumPy archives preserve bfloat16 bytes as void, without dtype metadata.
        if sample["board_float"].dtype.kind == "V":
            sample["board_float"] = sample["board_float"].view(jnp.bfloat16)
        generation = (json.loads((args.checkpoint / "LATEST").read_text())["generation"]
                      if args.checkpoint.is_dir() else str(args.checkpoint))
    else:
        generation, stats, sample = replay_sample(args.checkpoint)
    np.savez_compressed(args.output / "sample.npz", **sample)
    print(json.dumps({"replay": stats}), flush=True)
    model, params, config = load_model(args.checkpoint)
    if args.checkpoint.is_dir():
        assert json.loads((args.checkpoint / "LATEST").read_text())["generation"] == generation
    from nanoalphazero.muzero_config import normalize_exploration_config
    config = normalize_exploration_config(config)
    assert config["muzero_exploration_losses"] == "none"
    assert not config.get("exp_self_consistency", False)

    @jax.jit
    def unroll(params, boards, floats, actions, first_pad):
        obs = combine_observation_vmap(boards, floats.reshape(-1, 8, 8, 2))
        latent, logits, v = model.apply({"params": params}, obs, method=model.initial)
        absorbing = jnp.zeros_like(latent)
        values, rewards = [v], []
        for k in range(5):
            latent, r, logits, v = model.apply({"params": params}, latent, actions[:, k], method=model.recurrent)
            absorbing = jnp.where((first_pad == k+1)[:, None, None, None], latent, absorbing)
            values.append(v); rewards.append(r)
        return jnp.stack(values, 1), jnp.stack(rewards, 1), absorbing

    padding = sample["value_mask"] & ~sample["policy_mask"]
    first_pad = padding.argmax(-1)
    outputs = []
    for start in range(0, 256, 32):
        sl = slice(start, start+32)
        outputs.append(jax.device_get(unroll(params, sample["board_bool"][sl], sample["board_float"][sl],
                                            sample["action"][sl], first_pad[sl])))
    values, rewards, latents = (np.concatenate([x[k] for x in outputs]) for k in range(3))
    np.testing.assert_array_equal(sample["reward"][padding], 0)
    reward_padding = padding[:, :-1] & sample["reward_mask"]
    np.testing.assert_array_equal(sample["transition_reward"][reward_padding], 0)
    results = dict(replay=stats, generation=generation, checkpoint=str(args.checkpoint),
                   sample_from=str(args.sample_from) if args.sample_from else None,
                   recorded_padding_value=describe(values[padding]),
                   recorded_padding_reward=describe(rewards[reward_padding]),
                   real_value_error=describe((values-sample["reward"])[sample["policy_mask"] & sample["value_mask"]]))
    print(json.dumps(results), flush=True)

    @jax.jit
    def branches(params, latent, random_actions):
        logits, _ = model.apply({"params": params}, latent, method=lambda m, x: m.prediction(x))
        greedy = logits.argmax(-1)
        actions = jnp.concatenate((greedy[:, None], random_actions), 1)
        expanded = jnp.repeat(latent, actions.shape[1], axis=0)
        _, r, _, v = model.apply({"params": params}, expanded, actions.reshape(-1), method=model.recurrent)
        return r.reshape(actions.shape), v.reshape(actions.shape)
    random_actions = np.random.default_rng(7).integers(0, 4672, (128, 16), dtype=np.int32)
    outputs = []
    for start in range(0, 128, 4):
        outputs.append(jax.device_get(branches(params, latents[start:start+4], random_actions[start:start+4])))
    r, v = (np.concatenate([x[k] for x in outputs]) for k in range(2))
    results.update(greedy_value=describe(v[:, 0]), random_value=describe(v[:, 1:]),
                   greedy_reward=describe(r[:, 0]), random_reward=describe(r[:, 1:]),
                   greedy_worse_fraction=float((np.abs(v[:, 0]) > np.abs(v[:, 1:]).mean(1)).mean()))
    in_horizon = first_pad[:128] < 5
    results["branches_within_training_horizon"] = dict(
        windows=int(in_horizon.sum()), greedy_value=describe(v[in_horizon, 0]),
        random_value=describe(v[in_horizon, 1:]))
    np.savez_compressed(args.output / "predictions.npz", values=values, rewards=rewards,
                        branch_values=v, branch_rewards=r)
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
