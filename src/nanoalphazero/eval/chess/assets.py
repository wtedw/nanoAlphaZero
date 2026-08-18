"""Fetch and verify the small assets needed by standalone tournaments."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    sha256: str
    relative_path: str
    archive: str | None = None
    download_size_bytes: int | None = None
    extracted_size_bytes: int | None = None


ECO_OPENINGS = Asset(
    name="eco_openings",
    url="https://storage.googleapis.com/searchless_chess/data/eco_openings.pgn",
    sha256="17127332f58c2a47505833af349d7fa0e3c2afdf5e5a94993c2f0ef4a55ae625",
    relative_path="data/eval/eco_openings.pgn",
)
SEARCHLESS_9M = Asset(
    name="searchless_9m",
    url="https://storage.googleapis.com/searchless_chess/checkpoints/9M.zip",
    sha256="e7a62f07c14819ec77433f108d3490599cdf097a49afe64f1be3973d19c34eed",
    relative_path="artifacts/searchless_chess",
    archive="searchless_zip",
    download_size_bytes=66_375_867,
    extracted_size_bytes=66_487_444,
)
SEARCHLESS_136M = Asset(
    name="searchless_136m",
    url="https://storage.googleapis.com/searchless_chess/checkpoints/136M.zip",
    sha256="6161d9ab24a7b739a418a1634dd5709733d2b6ebd3a0c2cb6f3ad94f0d649157",
    relative_path="artifacts/searchless_chess",
    archive="searchless_zip",
    download_size_bytes=1_012_357_690,
    extracted_size_bytes=1_012_316_133,
)
SEARCHLESS_270M = Asset(
    name="searchless_270m",
    url="https://storage.googleapis.com/searchless_chess/checkpoints/270M.zip",
    sha256="96adba9d943fdf5189e3d8cfb15a84ae1a3904b9453d1e8f64e4b1a32decff27",
    relative_path="artifacts/searchless_chess",
    archive="searchless_zip",
    download_size_bytes=2_011_006_753,
    extracted_size_bytes=2_010_889_548,
)
STOCKFISH_16 = Asset(
    name="stockfish_16",
    url=(
        "https://github.com/official-stockfish/Stockfish/releases/download/"
        "sf_16/stockfish-ubuntu-x86-64-avx2.tar"
    ),
    sha256="9a461f249ccf64689706782b12e0d00e47cb9474b67ea77f2ba1a66b1e793b17",
    relative_path="artifacts/stockfish/16",
    archive="stockfish_binary_tar",
    download_size_bytes=41_594_880,
    extracted_size_bytes=40_442_144,
)

DEFAULT_ASSETS = (ECO_OPENINGS, SEARCHLESS_9M)
ASSETS = (*DEFAULT_ASSETS, STOCKFISH_16, SEARCHLESS_136M, SEARCHLESS_270M)

BAYESELO = Asset(
    name="bayeselo",
    url="https://www.remi-coulom.fr/Bayesian-Elo/bayeselo.tar.bz2",
    sha256="95f3d32381932ba9ccd1ab221ff33273cd3e48abedcfeba4c4f3f5a9b303a2a4",
    relative_path="artifacts/bayeselo",
    archive="bayeselo_source",
)

ASSETS_BY_NAME = {asset.name: asset for asset in (*ASSETS, BAYESELO)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe archive member {member.filename!r}")
        zip_file.extractall(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:bz2") as tar_file:
        for member in tar_file.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe archive member {member.name!r}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name!r}")
        tar_file.extractall(destination)


def _extract_stockfish_binary(archive: Path, destination: Path) -> None:
    member_name = "stockfish/stockfish-ubuntu-x86-64-avx2"
    with tarfile.open(archive, "r:") as tar_file:
        member = tar_file.getmember(member_name)
        if not member.isfile():
            raise ValueError(f"Stockfish archive member is not a file: {member_name}")
        source = tar_file.extractfile(member)
        if source is None:
            raise ValueError(f"could not read Stockfish archive member: {member_name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)
        destination.chmod(0o755)


def _archive_result(asset: Asset, target: Path) -> Path:
    if asset.archive == "searchless_zip":
        return target / Path(asset.url).stem
    if asset.archive == "bayeselo_source":
        return target / "BayesElo" / "bayeselo"
    if asset.archive == "stockfish_binary_tar":
        return target / "stockfish"
    raise ValueError(f"unknown archive type {asset.archive!r}")


def _source_marker(asset: Asset, target: Path) -> Path:
    if asset.archive == "searchless_zip":
        return target / f".{asset.name}.source.sha256"
    return target / ".source.sha256"


def _source_marker_matches(asset: Asset, target: Path) -> bool:
    marker = _source_marker(asset, target)
    if marker.exists() and marker.read_text().strip() == asset.sha256:
        return True
    # Accept the marker written by releases before multiple models were managed.
    legacy = target / ".source.sha256"
    return (
        asset.name == "searchless_9m"
        and legacy.exists()
        and legacy.read_text().strip() == asset.sha256
    )


def verified_archive_source_sha256(result: Path) -> str | None:
    """Return the verified source archive hash for a managed extracted asset."""
    result = result.resolve()
    for asset in ASSETS:
        if asset.archive != "searchless_zip":
            continue
        target = result.parent
        if _archive_result(asset, target).resolve() != result:
            continue
        return asset.sha256 if _source_marker_matches(asset, target) else None
    return None


def _ensure_model_download_space(asset: Asset, target: Path) -> None:
    if asset.archive != "searchless_zip":
        return
    if asset.download_size_bytes is None or asset.extracted_size_bytes is None:
        raise ValueError(f"missing size metadata for {asset.name}")
    headroom = 512 * 1024**2
    required = asset.download_size_bytes + asset.extracted_size_bytes + headroom
    free = shutil.disk_usage(target.parent).free
    if free < required:
        gib = 1024**3
        raise OSError(
            f"insufficient disk space for {asset.name}: need at least "
            f"{required / gib:.2f} GiB free for download and extraction, "
            f"but only {free / gib:.2f} GiB is available at {target.parent}"
        )


def _archive_is_ready(asset: Asset, result: Path) -> bool:
    if asset.archive == "searchless_zip":
        return (result / "6400000" / "params").exists()
    if asset.archive == "bayeselo_source":
        return result.is_file() and bool(result.stat().st_mode & 0o111)
    if asset.archive == "stockfish_binary_tar":
        return result.is_file() and bool(result.stat().st_mode & 0o111)
    return False


def fetch_asset(asset: Asset, root: Path) -> Path:
    target = root / asset.relative_path
    if not asset.archive and target.exists() and sha256_file(target) == asset.sha256:
        print(f"verified {asset.name}: {target}")
        return target
    marker = _source_marker(asset, target)
    result = _archive_result(asset, target) if asset.archive else target
    if (
        asset.archive
        and _archive_is_ready(asset, result)
        and _source_marker_matches(asset, target)
    ):
        print(f"found {asset.name}: {result}")
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    _ensure_model_download_space(asset, target)
    with tempfile.TemporaryDirectory(
        prefix="nanoaz-assets-", dir=target.parent
    ) as temp_dir:
        download = Path(temp_dir) / Path(asset.url).name
        print(f"downloading {asset.name} from {asset.url}", flush=True)
        with urllib.request.urlopen(asset.url) as response, download.open("wb") as out:
            shutil.copyfileobj(response, out)
        actual = sha256_file(download)
        if actual != asset.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {asset.name}: expected {asset.sha256}, got {actual}"
            )
        if asset.archive == "searchless_zip":
            target.mkdir(parents=True, exist_ok=True)
            _safe_extract(download, target)
            marker.write_text(asset.sha256 + "\n")
            return result
        if asset.archive == "bayeselo_source":
            make = shutil.which("make")
            if not make:
                raise FileNotFoundError(
                    "building BayesElo requires `make` and a C++ compiler; on "
                    "Debian/Ubuntu run `sudo apt-get install build-essential`"
                )
            extracted = Path(temp_dir) / "source"
            extracted.mkdir()
            _safe_extract_tar(download, extracted)
            source = extracted / "BayesElo"
            try:
                subprocess.run([make, "bayeselo"], cwd=source, check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    "failed to build BayesElo; install a C++ compiler (for "
                    "Debian/Ubuntu: `sudo apt-get install build-essential`)"
                ) from error
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target / "BayesElo", dirs_exist_ok=True)
            marker.write_text(asset.sha256 + "\n")
            return target / "BayesElo" / "bayeselo"
        if asset.archive == "stockfish_binary_tar":
            _extract_stockfish_binary(download, result)
            marker.write_text(asset.sha256 + "\n")
            return result
        download.replace(target)
    return target


def verify_asset(asset: Asset, root: Path) -> Path:
    target = root / asset.relative_path
    if asset.archive:
        result = _archive_result(asset, target)
        expected = (
            result / "6400000" / "params"
            if asset.archive == "searchless_zip"
            else result
        )
        if not expected.exists():
            raise FileNotFoundError(f"missing extracted asset {asset.name}: {expected}")
        if asset.archive in {"bayeselo_source", "stockfish_binary_tar"}:
            if not expected.stat().st_mode & 0o111:
                raise PermissionError(f"asset is not executable: {expected}")
        marker = _source_marker(asset, target)
        if not _source_marker_matches(asset, target):
            cli_name = asset.name.replace("_", "-")
            raise ValueError(
                f"missing or invalid source checksum marker for {asset.name}: {marker}; "
                f"run `uv run assets fetch {cli_name}` to re-download and verify "
                "the archive"
            )
        print(f"verified {asset.name}: {expected}")
        return result
    actual = sha256_file(target)
    if actual != asset.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {asset.name}: expected {asset.sha256}, got {actual}"
        )
    print(f"verified {asset.name}: {target}")
    return target


def _select_assets(targets: list[str] | None) -> tuple[Asset, ...]:
    if not targets:
        return DEFAULT_ASSETS
    normalized = [target.replace("-", "_") for target in targets]
    if "all" in normalized:
        if len(normalized) != 1:
            raise ValueError("`all` cannot be combined with individual asset names")
        return (*ASSETS, BAYESELO)
    return tuple(ASSETS_BY_NAME[target] for target in normalized)


def assets_main(
    action: str, root: str | None = None, targets: list[str] | None = None
) -> None:
    destination = Path(root).expanduser().resolve() if root else Path.cwd().resolve()
    for asset in _select_assets(targets):
        if action == "fetch":
            fetch_asset(asset, destination)
        else:
            verify_asset(asset, destination)
