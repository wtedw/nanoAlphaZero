"""MuZero host logging and local Trackio continuation, independent of training state."""

from dataclasses import dataclass


@dataclass
class TrainingLogger:
    run: object
    metadata: dict
    last_step: int = -1

    def log(self, metrics, *, step):
        # A crash may follow logging but precede the next checkpoint. Recomputed
        # cycles still train, but do not append duplicate rows to this run.
        if step > self.last_step:
            self.run.log(metrics, step=step)
            self.last_step = step

    def finish(self):
        self.run.finish()


def make_muzero_logger(config, *, resume=None, trackio_run=None):
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
        return TrainingLogger(run, metadata, last_step)
    if config.get("enable_wandb", False):
        import wandb
        return TrainingLogger(wandb.init(project="nanoAlphaZero-muzero", config=config), {})
    return None
