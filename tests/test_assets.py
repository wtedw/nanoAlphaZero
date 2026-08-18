from pathlib import Path

import pytest

from nanoalphazero.eval.chess import assets


def test_default_selection_preserves_existing_assets():
    assert assets._select_assets(None) == assets.DEFAULT_ASSETS
    assert assets.SEARCHLESS_136M not in assets._select_assets(None)
    assert assets.SEARCHLESS_270M not in assets._select_assets(None)
    assert assets.STOCKFISH_16 not in assets._select_assets(None)
    assert assets.BAYESELO not in assets._select_assets(None)


def test_selection_accepts_cli_names_and_all():
    assert assets._select_assets(["searchless-9m"])[0].name == "searchless_9m"
    assert assets._select_assets(["searchless-136m"]) == (
        assets.SEARCHLESS_136M,
    )
    assert assets._select_assets(["searchless-270m"]) == (
        assets.SEARCHLESS_270M,
    )
    assert assets._select_assets(["bayeselo"]) == (assets.BAYESELO,)
    assert assets._select_assets(["stockfish-16"]) == (assets.STOCKFISH_16,)
    assert assets._select_assets(["all"]) == (*assets.ASSETS, assets.BAYESELO)


def test_searchless_archive_result_uses_model_name(tmp_path):
    target = tmp_path / "artifacts" / "searchless_chess"

    assert assets._archive_result(assets.SEARCHLESS_136M, target) == target / "136M"
    assert assets._archive_result(assets.SEARCHLESS_270M, target) == target / "270M"
    assert assets._source_marker(assets.SEARCHLESS_136M, target) != (
        assets._source_marker(assets.SEARCHLESS_270M, target)
    )


def test_stockfish_archive_result_is_versioned_binary(tmp_path):
    target = tmp_path / "artifacts" / "stockfish" / "16"

    assert assets._archive_result(assets.STOCKFISH_16, target) == target / "stockfish"


def test_verified_archive_source_sha256_uses_model_specific_marker(tmp_path):
    target = tmp_path / "artifacts" / "searchless_chess"
    model_9m = assets._archive_result(assets.SEARCHLESS_9M, target)
    model_136m = assets._archive_result(assets.SEARCHLESS_136M, target)
    model_9m.mkdir(parents=True)
    model_136m.mkdir()
    assets._source_marker(assets.SEARCHLESS_9M, target).write_text(
        assets.SEARCHLESS_9M.sha256 + "\n"
    )
    assets._source_marker(assets.SEARCHLESS_136M, target).write_text(
        assets.SEARCHLESS_136M.sha256 + "\n"
    )

    assert assets.verified_archive_source_sha256(model_9m) == (
        assets.SEARCHLESS_9M.sha256
    )
    assert assets.verified_archive_source_sha256(model_136m) == (
        assets.SEARCHLESS_136M.sha256
    )


def test_verified_archive_source_sha256_rejects_wrong_model_marker(tmp_path):
    target = tmp_path / "artifacts" / "searchless_chess"
    model_136m = assets._archive_result(assets.SEARCHLESS_136M, target)
    model_136m.mkdir(parents=True)
    (target / ".source.sha256").write_text(assets.SEARCHLESS_9M.sha256 + "\n")

    assert assets.verified_archive_source_sha256(model_136m) is None


def test_searchless_download_fails_before_transfer_when_disk_is_too_small(
    monkeypatch, tmp_path
):
    usage = type("Usage", (), {"free": 1})()
    monkeypatch.setattr(assets.shutil, "disk_usage", lambda path: usage)
    monkeypatch.setattr(
        assets.urllib.request,
        "urlopen",
        lambda url: pytest.fail("download started before disk-space check"),
    )

    with pytest.raises(OSError, match="insufficient disk space for searchless_270m"):
        assets.fetch_asset(assets.SEARCHLESS_270M, tmp_path)


def test_all_cannot_be_combined_with_specific_asset():
    with pytest.raises(ValueError, match="cannot be combined"):
        assets._select_assets(["all", "bayeselo"])


def test_verify_managed_bayeselo(tmp_path):
    binary = tmp_path / "artifacts" / "bayeselo" / "BayesElo" / "bayeselo"
    binary.parent.mkdir(parents=True)
    binary.write_text("binary")
    binary.chmod(0o755)
    marker = binary.parents[1] / ".source.sha256"
    marker.write_text(assets.BAYESELO.sha256 + "\n")

    assert assets.verify_asset(assets.BAYESELO, tmp_path) == binary


def test_safe_tar_rejects_parent_traversal(monkeypatch, tmp_path):
    class UnsafeMember:
        name = "../escape"

        def issym(self):
            return False

        def islnk(self):
            return False

    class FakeTar:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def getmembers(self):
            return [UnsafeMember()]

    monkeypatch.setattr(assets.tarfile, "open", lambda *args, **kwargs: FakeTar())
    with pytest.raises(ValueError, match="unsafe archive member"):
        assets._safe_extract_tar(Path("archive"), tmp_path)
