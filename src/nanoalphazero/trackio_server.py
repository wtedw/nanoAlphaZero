"""Serve the shared training dashboard without importing the JAX runtime."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--project", default="nanoAlphaZero-muzero")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(os.environ.get("TRACKIO_DIR", "artifacts/trackio")),
        help="Run directory (default: TRACKIO_DIR or artifacts/trackio).",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    os.environ["TRACKIO_DIR"] = str(args.data_dir.expanduser().resolve())

    try:
        import trackio
    except ModuleNotFoundError as exc:
        if exc.name != "trackio":
            raise
        parser.error("Trackio is optional; use uv run --extra trackio trackio-server")

    print(f"Trackio data: {os.environ['TRACKIO_DIR']}", flush=True)
    trackio.show(
        project=args.project,
        host=args.host,
        server_port=args.port,
        open_browser=False,
        block_thread=True,
    )


if __name__ == "__main__":
    main()
