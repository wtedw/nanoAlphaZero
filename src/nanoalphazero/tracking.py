"""Experiment tracking and local Trackio continuation, independent of training state."""

from dataclasses import dataclass, field
import time
import sys


@dataclass
class TrainingTracker:
    run: object | None = None
    metadata: dict = field(default_factory=dict)
    last_step: int = -1
    resume_step: int = field(init=False)

    def __post_init__(self):
        self.resume_step = self.last_step

    def should_log(self, cycle):
        """Allow new cycles, including multiple writes within the same cycle."""
        return self.run is not None and cycle > self.resume_step

    def log_training(self, metrics, *, step, tables=None, histograms=None, figures=None):
        """Record training results, leaving room for later evaluation metrics."""
        if not self.should_log(step):
            return
        media = {}
        if tables or histograms or figures:
            backend = "trackio" if "trackio" in self.metadata else "wandb"
            media = media_logs(tables or {}, histograms or {}, figures or {}, backend)
        self.log({**metrics, **media}, step=step, commit=False)

    def log(self, metrics, *, step, commit=True, html=None):
        if not self.should_log(step):
            return
        # A crash may follow logging but precede the next checkpoint. Recomputed
        # cycles still train, but do not append duplicate rows to this run.
        # Multiple calls may add different metrics at the current cycle. Only
        # cycles persisted by the previous process are skipped wholesale.
        if html:
            import importlib
            backend = importlib.import_module("trackio" if "trackio" in self.metadata else "wandb")
            metrics = {**metrics, **{name: backend.Html(document, inject=False)
                                    for name, document in html.items()}}
        if "trackio" in self.metadata:
            if metrics:
                self.run.log(metrics, step=step)
        else:
            # W&B requires the training row to remain open until later
            # checkpoint/evaluation metrics for this step are available.
            self.run.log(metrics, step=step, commit=commit)
        self.last_step = max(self.last_step, step)

    def log_files(self, name, files, *, step, kind="diagnostics", metadata=None):
        """Publish explicit local files through the active backend.

        This never discovers or uploads checkpoints. Callers choose each file.
        """
        if not self.should_log(step) or not files:
            return
        import importlib
        backend = importlib.import_module("trackio" if "trackio" in self.metadata else "wandb")

        artifact = backend.Artifact(
            f"{name}-{self.run.id}", type=kind, metadata={"cycle": step, **(metadata or {})},
        )
        for filename, path in files.items():
            artifact.add_file(str(path), name=filename)
        self.run.log_artifact(artifact, aliases=["latest", f"cycle-{step}"])

    def flush(self, timeout=30.0):
        """Best-effort wait for local Trackio's writer; never block saving on failure.

        Trackio has no public flush API. Check the database acknowledgement rather
        than manipulating its private queues, which may already be in flight.
        W&B remains asynchronous; this wait applies only to local Trackio.
        """
        identity = self.metadata.get("trackio")
        if identity is None or self.last_step < 0:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                persisted = _persisted_step(identity, deadline)
            except Exception as exc:
                print(f"Trackio persistence check failed: {exc}; continuing with checkpoint save",
                      file=sys.stderr, flush=True)
                return
            if persisted is not None and persisted >= self.last_step:
                return
            time.sleep(max(0, min(0.05, deadline - time.monotonic())))
        print(f"Trackio has not persisted training cycle {self.last_step} after {timeout:g}s; "
              "continuing with checkpoint save", file=sys.stderr, flush=True)

    def finish(self):
        if self.run is not None:
            self.run.finish()


def _persisted_step(identity, deadline):
    """Read local acknowledgement with bounded SQLite locking and query time.

    Trackio's public reader has a fixed connection timeout. Keep this small
    schema-dependent adapter here; schema/setup failures fall back to saving.
    """
    import sqlite3
    from contextlib import closing
    from trackio.sqlite_storage import SQLiteStorage

    path = SQLiteStorage.get_project_db_path(identity["project"])
    if not path.exists():
        return None
    uri = path.resolve().as_uri() + "?mode=ro"
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    with closing(sqlite3.connect(uri, uri=True, timeout=remaining)) as connection:
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        # A new project's file can appear before its writer creates the schema.
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metrics'"
        ).fetchone() is None:
            return None
        return connection.execute("SELECT MAX(step) FROM metrics WHERE run_id = ?",
                                  (identity["id"],)).fetchone()[0]


def make_tracker(config, *, resume=None, trackio_run=None):
    if config.get("enable_trackio", False):
        try:
            import trackio
        except ModuleNotFoundError as exc:
            if exc.name != "trackio":
                raise
            raise RuntimeError(
                "Trackio logging requires: uv run --extra trackio python muzero.py --enable-trackio"
            ) from exc
        from nanoalphazero.ckpt import checkpoint_metadata

        saved = checkpoint_metadata(resume).get("trackio", {}) if resume else {}
        project = saved.get("project", "nanoAlphaZero-muzero")
        name = trackio_run or saved.get("name")
        options = dict(name=name, resume="must") if name else {}
        run = trackio.init(project=project, config=config, **options)
        if saved and not trackio_run and run.id != saved["id"]:
            run.finish()
            raise ValueError("Trackio run name now refers to a different run ID")
        last_step = -1
        if name:
            from trackio.sqlite_storage import SQLiteStorage
            previous = SQLiteStorage.get_max_step_for_run(project, run.name, run_id=run.id)
            last_step = previous if previous is not None else -1
        metadata = {"trackio": {"project": project, "name": run.name, "id": run.id}}
        print(f"Trackio run: {run.name} ({run.id}); last logged cycle: {last_step}", flush=True)
        return TrainingTracker(run, metadata, last_step)
    if config.get("enable_wandb", False):
        import wandb
        return TrainingTracker(wandb.init(project="nanoAlphaZero-muzero", config=config), {})
    return TrainingTracker()


def media_logs(tables, histograms, figures, backend):
    """Wrap ordinary media for one backend, without knowing their game or origin."""
    import importlib

    tracker = importlib.import_module(backend)
    logs = {name: tracker.Table(**table) for name, table in tables.items()}
    logs.update({name: tracker.Histogram(values) for name, values in histograms.items()})
    for name, figure in figures.items():
        logs[name] = tracker.Plotly(figure) if backend == "wandb" else tracker.Html(figure)
    return logs
