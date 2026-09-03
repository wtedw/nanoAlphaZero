"""Reproducible standalone Hex evaluation orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from nanoalphazero.eval.hex.config import (
    config_hash,
    load_config,
    public_config,
    resolve_path,
)
from nanoalphazero.eval.hex.engine import (
    MoHexPool,
    resolve_config as resolve_engine_config,
    resolve_executable,
)


SCHEDULER_VERSION = "hex_mohex_v1"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_engine_value(config: dict, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return str(resolve_path(config, value))
    return value


def _prepare_run(config, source: Path, resume, output_root):
    digest = config_hash(config)
    if resume:
        run_dir = Path(resume).expanduser().resolve()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("scheduler_version") != SCHEDULER_VERSION:
            raise ValueError("cannot resume an incompatible Hex evaluation")
        if manifest.get("config_hash") != digest:
            raise ValueError("resume config does not match the existing run")
        return run_dir, manifest
    root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else resolve_path(config, config["hex_eval"].get("output_root", "runs"))
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}-{digest[:10]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, run_dir / "config.toml")
    manifest = {
        "scheduler_version": SCHEDULER_VERSION,
        "config_hash": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_units": [],
    }
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir, manifest


def _write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    temporary.replace(path)


def _write_grid(path: Path, records: list[dict], size: int) -> None:
    cell = 72
    margin = 44
    width = margin * 2 + cell * size
    colors = {0: "#2979ff", 1: "#e53935", None: "#777777"}
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{width}" viewBox="0 0 {width} {width}">',
        '<rect width="100%" height="100%" fill="#111827"/>',
        f'<text x="{width / 2}" y="26" fill="white" text-anchor="middle" '
        f'font-family="sans-serif">Hex {size}x{size}: model P1 vs MoHex</text>',
    ]
    for record in records:
        action = int(record["opening_action"])
        row, column = divmod(action, size)
        x = margin + column * cell
        y = margin + row * cell
        pieces.append(
            f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" '
            f'fill="{colors[record["winner"]]}"/>'
        )
        pieces.append(
            f'<text x="{x + cell / 2 - 1}" y="{y + cell / 2 + 6}" '
            f'fill="white" text-anchor="middle" font-family="monospace">'
            f'{record["opening_vertex"]} · {record["plies"]}</text>'
        )
    pieces.append("</svg>\n")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(pieces))
    temporary.replace(path)


def run_evaluation(config, source: Path, pool, *, resume=None, output_root=None):
    run_dir, manifest = _prepare_run(config, source, resume, output_root)
    summary_path = run_dir / "summary.json"
    if "all_first_moves" in manifest["completed_units"] and summary_path.is_file():
        print(f"already complete: {run_dir}")
        return run_dir, json.loads(summary_path.read_text())

    # Import JAX-dependent modules only after the MoHex process bank exists.
    import jax

    from nanoalphazero import core
    from nanoalphazero.checkpoint import apply_checkpoint_model_config, load_checkpoint
    from nanoalphazero.config import get_hex_config
    from nanoalphazero.eval.hex.runtime import print_grid, run_match

    size = int(config["hex_eval"]["board_size"])
    checkpoint = resolve_path(config, config["model"]["checkpoint"])
    params, model_metadata = load_checkpoint(str(checkpoint))
    expected_env = f"hexnoswap_{size}x{size}"
    if model_metadata.get("env_id") != expected_env:
        raise ValueError(
            f"checkpoint environment {model_metadata.get('env_id')!r} does not "
            f"match {expected_env!r}"
        )
    model_config = get_hex_config(size)
    env = core.make_env(model_config)
    model_config.update(game_obs_shape=env.obs_shape, game_num_actions=env.num_actions)
    model_config = apply_checkpoint_model_config(model_config, model_metadata)
    model, initialized = core.make_model(
        model_config, jax.random.PRNGKey(int(config["hex_eval"].get("seed", 1))),
        core.REPLICATED_SHARDING,
    )
    actual = jax.tree.map(lambda value: tuple(value.shape), params)
    expected = jax.tree.map(lambda value: tuple(value.shape), initialized["params"])
    if actual != expected:
        raise ValueError(f"checkpoint parameter shapes differ from Hex model: {checkpoint}")
    params = jax.tree.map(
        lambda value: jax.device_put(value, core.REPLICATED_SHARDING), params
    )
    run_mcts = core.make_mcts(
        model_config, env, model, data_sharding=core.DATA_PARALLEL_SHARDING
    )
    records, match_summary = run_match(
        run_mcts,
        env,
        model_config,
        params,
        pool,
        seed=int(config["hex_eval"].get("seed", 1)),
    )
    print_grid(records, size)
    _write_jsonl(run_dir / "games.jsonl", records)
    _write_grid(run_dir / "opening-grid.svg", records, size)
    resolved = copy.deepcopy(public_config(config))
    resolved["model"]["checkpoint"] = str(checkpoint)
    resolved["opponent"]["path"] = str(pool.executable)
    resolved["opponent"]["config"] = str(pool.config)
    _write_json(run_dir / "resolved.json", resolved)
    summary = {
        "scheduler_version": SCHEDULER_VERSION,
        "config_hash": config_hash(config),
        **match_summary,
        "checkpoint_sha256": _sha256(checkpoint),
        "mohex_sha256": _sha256(pool.executable),
        "mohex_config_sha256": _sha256(pool.config),
        "devices": [str(device) for device in jax.devices()],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(summary_path, summary)
    manifest["completed_units"] = ["all_first_moves"]
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(run_dir / "manifest.json", manifest)
    print(f"wrote {run_dir}")
    return run_dir, summary


def evaluation_main(config_path: str, *, resume=None, output_root=None):
    config, source = load_config(config_path)
    checkpoint = resolve_path(config, config["model"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Hex checkpoint not found: {checkpoint}")
    opponent = config["opponent"]
    engine_path = _resolved_engine_value(config, str(opponent["path"]))
    resolve_executable(engine_path)
    engine_config = opponent.get("config", "default")
    if engine_config != "default":
        engine_config = str(resolve_path(config, engine_config))
    resolve_engine_config(engine_config)

    run_dir, manifest = _prepare_run(config, source, resume, output_root)
    summary_path = run_dir / "summary.json"
    if "all_first_moves" in manifest["completed_units"] and summary_path.is_file():
        print(f"already complete: {run_dir}")
        return run_dir, json.loads(summary_path.read_text())

    size = int(config["hex_eval"]["board_size"])
    with MoHexPool(
        engine_path,
        engine_config,
        size,
        size * size,
        seed=int(config["hex_eval"].get("seed", 1)),
    ) as pool:
        return run_evaluation(
            config, source, pool, resume=str(run_dir), output_root=output_root
        )
