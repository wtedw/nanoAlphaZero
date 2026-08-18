import hashlib
from pathlib import Path

from nanoalphazero.eval.wandb_artifacts import (
    _artifact_target,
    fetch_wandb_checkpoint,
)


class FakeArtifact:
    qualified_name = "entity/project/model:v2"
    digest = "artifact-digest"

    def __init__(self):
        self.downloads = 0

    def download(self, root):
        self.downloads += 1
        destination = Path(root)
        (destination / "model.pkl").write_bytes(b"trusted checkpoint")
        return str(destination)

    def files(self):
        return []


class FakeApi:
    def __init__(self, artifact):
        self.value = artifact

    def artifact(self, ref):
        assert ref == "entity/project/model:v2"
        return self.value


def test_artifact_target_requires_pinned_version(tmp_path):
    target = _artifact_target(
        tmp_path, "entity/project/model:v2", "model.pkl"
    )
    assert target == tmp_path / "entity/project/model/v2/model.pkl"


def test_fetch_wandb_checkpoint_is_verified_and_cached(tmp_path):
    artifact = FakeArtifact()
    api = FakeApi(artifact)

    first = fetch_wandb_checkpoint(
        "entity/project/model:v2", "model.pkl", tmp_path, api=api
    )
    second = fetch_wandb_checkpoint(
        "entity/project/model:v2", "model.pkl", tmp_path, api=api
    )

    assert first == second
    assert first.read_bytes() == b"trusted checkpoint"
    assert artifact.downloads == 1
    assert hashlib.sha256(first.read_bytes()).hexdigest() in (
        first.parent / ".wandb-artifact.json"
    ).read_text()

