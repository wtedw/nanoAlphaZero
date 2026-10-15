"""Distributed startup, called before backend-sensitive package imports.

The launcher owns host selection and process supervision. This module only
initializes JAX; it never removes TPU locks or starts/stops worker processes.
"""

from dataclasses import dataclass


@dataclass
class DistributedArgs:
    """Opt-in pod discovery, or explicit topology for non-TPU smoke tests."""

    enabled: bool = False
    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    initialization_timeout: int = 120


def initialize_distributed(args: DistributedArgs):
    """Initialize before any device query, array creation, or core import.

    TPU pods normally need only ``enabled=True``. Explicit topology is useful
    for CPU tests and launchers without TPU runtime discovery. Every worker
    must receive the same coordinator/count and a distinct process ID.
    """
    explicit = (args.coordinator_address, args.num_processes, args.process_id)
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("Distributed coordinator, process count and ID must be supplied together")
        if args.num_processes < 1 or not 0 <= args.process_id < args.num_processes:
            raise ValueError("Distributed process ID must be within the process count")
        if not args.enabled:
            raise ValueError("Explicit distributed topology requires --distributed.enabled")
    if args.initialization_timeout <= 0:
        raise ValueError("Distributed initialization timeout must be positive")
    if not args.enabled:
        return
    import jax

    if jax.distributed.is_initialized():
        raise RuntimeError("Distributed JAX was already initialized before MuZero startup")
    kwargs = {"initialization_timeout": args.initialization_timeout}
    if args.coordinator_address is not None:
        kwargs.update(coordinator_address=args.coordinator_address,
                      num_processes=args.num_processes, process_id=args.process_id)
    jax.distributed.initialize(**kwargs)


def validate_muzero_run(config, process_count, *, checkpoint=False, hex_evaluation=False):
    """Reject features whose multihost publication/collectives are not ready yet."""
    if process_count == 1:
        return
    if not config.get("enable_sharding", False):
        raise ValueError("Multihost MuZero requires global sharding")
    unsupported = []
    if config.get("eval_period") or hex_evaluation:
        unsupported.append("ladder/MoHex evaluation (use --ladder-period 0)")
    if unsupported:
        raise ValueError("Multihost MuZero currently supports bounded training with these features disabled: "
                         + ", ".join(unsupported))
