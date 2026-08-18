"""Resolve pinned W&B checkpoint artifacts for tournament agents."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from nanoalphazero.eval.chess.assets import sha256_file
from nanoalphazero.eval.chess.config import load_config, resolve_path


def _artifact_target(root: Path, artifact_ref: str, filename: str) -> Path:
    parts = artifact_ref.split("/")
    if len(parts) != 3 or ":" not in parts[-1]:
        raise ValueError(
            "artifact_path must pin entity/project/name:version; "
            f"got {artifact_ref!r}"
        )
    name, version = parts[-1].rsplit(":", 1)
    if not all((*parts[:2], name, version)):
        raise ValueError(f"invalid artifact_path {artifact_ref!r}")
    relative_filename = Path(filename)
    if relative_filename.is_absolute() or ".." in relative_filename.parts:
        raise ValueError(f"invalid artifact_filename {filename!r}")
    return root.joinpath(*parts[:2], name, version, relative_filename)


def fetch_wandb_checkpoint(
    artifact_ref: str,
    filename: str,
    root: Path,
    *,
    api=None,
) -> Path:
    """Download one immutable W&B artifact version and record its identity."""
    target = _artifact_target(root.resolve(), artifact_ref, filename)
    marker = target.parent / ".wandb-artifact.json"
    if target.is_file() and marker.is_file():
        recorded = json.loads(marker.read_text())
        if (
            recorded.get("artifact") == artifact_ref
            and recorded.get("filename") == filename
            and recorded.get("sha256") == sha256_file(target)
        ):
            print(f"verified {artifact_ref}/{filename}: {target}")
            return target

    if api is None:
        import wandb

        api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    identity = {
        "artifact": artifact_ref,
        "qualified_name": artifact.qualified_name,
        "digest": artifact.digest,
        "filename": filename,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="nanoaz-wandb-", dir=target.parent
    ) as temp_dir:
        downloaded_root = Path(artifact.download(root=temp_dir))
        downloaded = downloaded_root / filename
        if not downloaded.is_file():
            available = sorted(file.name for file in artifact.files())
            raise FileNotFoundError(
                f"artifact {artifact_ref} has no {filename!r}; files: {available}"
            )
        temporary_target = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(downloaded, temporary_target)
        temporary_target.replace(target)
    identity["sha256"] = sha256_file(target)
    marker.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(f"downloaded {artifact_ref}/{filename}: {target}")
    return target


def materialize_agent_checkpoints(
    config: dict[str, Any], *, root: str | Path | None = None, api=None
) -> list[Path]:
    """Fetch configured W&B artifacts and attach resolved local checkpoints."""
    if root is None:
        configured_root = config["tournament"].get(
            "artifact_root", "../artifacts/wandb"
        )
        destination = resolve_path(config, configured_root)
    else:
        destination = Path(root).expanduser().resolve()
    resolved = []
    for agent in config["agents"]:
        artifact_ref = agent.get("artifact_path")
        if not artifact_ref:
            continue
        checkpoint = fetch_wandb_checkpoint(
            str(artifact_ref),
            str(agent["artifact_filename"]),
            destination,
            api=api,
        )
        agent["checkpoint"] = str(checkpoint)
        resolved.append(checkpoint)
    return resolved


def artifacts_main(config_path: str, root: str | None = None) -> None:
    config, _ = load_config(config_path)
    paths = materialize_agent_checkpoints(config, root=root)
    if not paths:
        print("no W&B artifacts configured")
