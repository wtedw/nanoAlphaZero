# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "chex==0.1.91",
#     "chess==1.11.2",
#     "dm-haiku==0.0.16",
#     "flashbax",
#     "flax==0.10.1",
#     "jax==0.8.1",
#     "libtpu==0.0.30; sys_platform == 'linux'",
#     "mctx",
#     "numpy==2.3.5",
#     "plotly>=6,<7",
#     "optax==0.2.7",
#     "orbax-checkpoint==0.11.30",
#     "pgx1",
#     "python-chess==1.999",
#     "safetensors==0.8.0",
#     "scipy==1.16.3",
#     "wandb==0.21.0",
# ]
# [tool.uv.sources]
# pgx1 = { git = "https://github.com/wtedw/pgx1.git", rev = "fa313c84338d93ab96fc02bc7c658364bf43098f" }
# mctx = { git = "https://github.com/wtedw/mctx", rev = "6cf1a39" }
# flashbax = { git = "https://github.com/instadeepai/flashbax.git", rev = "e0199d7bb232c622a19d3c28f9d6b34eb8215eab" }
# ///

# MIT License
#
# Copyright (c) 2026 Ted Wong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# --------------------------------------------------------------------------------
#
# Code in `src/nanoalphazero/mcts.py` is adapted (with modifications) from
# https://github.com/google-deepmind/mctx, used under the Apache License,
# Version 2.0. The adapted code carries the following notice from the original
# source files:
#
#     Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
# --------------------------------------------------------------------------------
#
# The neural-network architecture in `src/nanoalphazero/model.py` is adapted (with
# modifications) from KataGo:
# https://github.com/lightvector/KataGo
#
# The adapted code is used under the following KataGo MIT license:
#
# ----------------------------------------
#
# Copyright 2025 David J Wu ("lightvector") and/or other authors of the content in this repository.
# (See 'CONTRIBUTORS' file for a list of authors as well as other indirect contributors).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""MuZero: single-file implementation and running instructions.

GET STARTED
Copy only this file. Install uv and Python 3.11+; uv fetches the dependencies
declared above on first use (network access and Git are needed for that fetch).
You do not need a repository checkout, package installation, or TOML config.
Run these commands in the directory containing muzero.py:

  uv run muzero.py train --preset smoke-cpu       # Tiny four-device CPU smoke
  uv run muzero.py train --preset smoke-tpu       # Tiny four-device TPU smoke
  uv run muzero.py train --env hex4               # Full Hex 4 training on TPU
  uv run muzero.py train --env hex5               # Full Hex 5 training on TPU
  uv run muzero.py train                         # Same as --env hex4

FULL TRAINING DEFAULTS
TPU, four devices, spatial h/g/f networks, staged sequence replay, seed 0, and
AlphaZero's per-environment network size, batch sizes, learning rate and update
schedule. Hex 5 uses training batch 4,096. W&B is enabled for full training;
authenticate before launching. Each run gets a unique descriptive W&B name.
Only one JAX process may own the TPU; finish/stop your run before starting another.
Smoke presets use tiny settings and disable W&B. CPU smokes validate behavior,
not TPU performance. CPU mode supplies four virtual devices unless XLA_FLAGS
already specifies a count. Full training defaults are large, including on CPU.

OUTPUT AND CHECKPOINTS
Output defaults to a new artifacts/muzero-ENV-TIMESTAMP-ID directory, printed
when training starts. Override it with --output DIR (DIR must not exist).
Runs retain resolved config, manifest, script snapshot and metrics.jsonl.
Checkpoints are opt-in; --save saves the final checkpoint, and
--checkpoint-period 50 also saves every 50 cycles. They are not W&B artifacts.

  uv run muzero.py train --env hex4 --save --checkpoint-period 50
  uv run muzero.py train --env hex4 --save --checkpoint-period 50 \
    --resume artifacts/previous-run/cycle-000050.safetensors

Resume with the same settings as the interrupted run; resolved configuration
must match exactly. Keep heldout-initial-selfplay.npz beside its checkpoint when
held-out evaluation was enabled. Use --stop-file PATH to request a graceful stop
at a cycle boundary by creating PATH while the run is active.

SETTINGS WITHOUT TOML
  uv run muzero.py train --env hex5 --seed 2 --learning-rate 0.001 --save
  uv run muzero.py train --env hex4 --cycles 5 --no-wandb
  uv run muzero.py train --env hex6 --remat-blocks --remat-unroll --save
  uv run muzero.py train --env hex5 --print-config
  uv run muzero.py train --help

Other flags include --network vector|spatial, --width, --depth, --unroll,
--train-batch-size, --selfplay-batch-size, --eval-period and --platform cpu|tpu.
--remat-blocks and --remat-unroll recompute activations to reduce training memory.
More than ten cycles require W&B; use --no-wandb only for short checks.
Environments: ttt, connect4, hex4 through hex9, go3 through go9, chess.
Compatible interfaces do not imply strong play; Hex has the most training evidence.
An existing TOML can still be supplied for advanced experiments. Precedence is
defaults/preset, optional TOML, explicit CLI flags. With a TOML and no preset,
the original package's TOML defaults apply for backward compatibility.

EVALUATION
  uv run muzero.py eval CHECKPOINT --platform cpu --output /tmp/muzero-eval-new
  uv run muzero.py eval CHECKPOINT --alphazero AZ_CHECKPOINT --output artifacts/eval-new
  uv run muzero.py train --env hex4 --hex-eval-period 50 \
    --hex-eval-engine-path /path/to/mohex --save

Evaluation compares policy-only and learned-search play against each other and
random play. --alphazero adds a supplied reference checkpoint. For MoHex, supply
--mohex-engine-path to eval, or --hex-eval-engine-path with --hex-eval-period to
train. The engine binary is external; its default configuration is embedded.
MoHex evaluation is off by default so training needs no external engine.
Use --platform tpu for TPU evaluation (the eval default); never overlap TPU jobs.
Use `uv run muzero.py eval --help` for all evaluation options.

IMPLEMENTATION AND MAINTENANCE
Actual self-play -> staging -> consume/drain -> contiguous replay -> unrolled
h/g/f losses. The root uses real observations and legality; both hypothetical
search rungs use learned dynamics only. Network depth, search budget and training
unroll length are separate. Reference engines/tables are evaluation-only.
The package remains authoritative. From its root, regenerate the readable export:
  uv run evals/muzero/export_flat.py
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    uv run pytest tests/test_muzero_standalone.py
There are no embedded Python modules or import loaders. Package equivalence,
copied-script training and checkpoint loading are CPU-tested; this export does
not establish new playing-strength or TPU performance results.
"""

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib

STANDALONE_PRESETS = {'smoke-cpu': {'env': 'ttt',
               'defaults': 'pilot_v1',
               'platform': 'cpu',
               'devices': 4,
               'width': 16,
               'depth': 1,
               'selfplay_batch_size': 8,
               'train_batch_size': 8,
               'replay_batches': 2,
               'cycles': 2,
               'updates_per_cycle': 2,
               'data_pipeline': 'staged',
               'staging_batches': 2,
               'consume_size': 8,
               'replay_positions': 128,
               'replay_warmup_cycles': 1},
 'smoke-tpu': {'env': 'hex4',
               'defaults': 'pilot_v1',
               'platform': 'tpu',
               'devices': 4,
               'width': 32,
               'depth': 1,
               'selfplay_batch_size': 32,
               'train_batch_size': 32,
               'replay_batches': 2,
               'cycles': 2,
               'updates_per_cycle': 2},
 'hex4-staged-smoke': {'env': 'hex4',
                       'defaults': 'alphazero',
                       'data_pipeline': 'staged',
                       'network': 'spatial',
                       'width': 16,
                       'depth': 1,
                       'selfplay_batch_size': 32,
                       'train_batch_size': 32,
                       'consume_size': 32,
                       'replay_positions': 1024,
                       'replay_warmup_cycles': 2,
                       'cycles': 2,
                       'updates_per_cycle': 2,
                       'eval_period': 0,
                       'diagnostic_period': 1,
                       'checkpoint_period': 1,
                       'hex_eval_period': 1}}
STANDALONE_SOURCE = {'git_commit': '0bf30000cdaaee23ddcd6c457e813271a539f063',
 'source_sha256': {'src/nanoalphazero/config.py': '0dd387912de774d77a111fefadb1bf310a84408d17fe2ea48ffc451f53452426',
                   'src/nanoalphazero/buffers.py': '61eb1102f77ccdb0d4e878be9e560d0e8aec0038d75e32c2e0c23d2f68c71588',
                   'src/nanoalphazero/model.py': '700ea6a9d503196abb1778c47a7523a293ac32ad57fde4e69d3b21839fb91bbd',
                   'src/nanoalphazero/mcts.py': '2b42f2204227539d64858ee0521ef781a0c9c0ea764fa2205887358e908fa9cc',
                   'src/nanoalphazero/checkpoint.py': '435047097207e8c891c92bf38b9387888206fcbe7c9fe48584eabe5f9f857221',
                   'src/nanoalphazero/core.py': 'c8adb2b7692dfc1de037ad0a082cd26d9e211d2b183def5b7074f943ed9be1c4',
                   'src/nanoalphazero/eval/hex/perfect_play.py': '6b91c309ecc5ec8b98082371bba8ba156d3ffebe9fc21ef41a571feb440ede0f',
                   'src/nanoalphazero/eval/hex/engine.py': 'bc4e01e200271c5f1d9c6424b296c2b545c84c99eaa34d4cebf002cdedf0bd62',
                   'src/nanoalphazero/eval/hex/runtime.py': 'bec5ca703f708d670b70737d1a673a2f061f20678577ebe6fafca4418da774ee',
                   'src/nanoalphazero/eval/hex/training.py': 'd664401dae051813eca72b9c6b8a8b66852e7932015e1ca5d90d28a31c0a2bb5',
                   'src/nanoalphazero/training.py': '028056278e178a4ed36eaf5e2e2dd9cc4d15c22c66eb0da3206715710335aecb',
                   'src/nanoalphazero/research/muzero/model.py': '68db09152ae592025d2cba55009d4f087ee03d189a73cc4cd18049d934959722',
                   'src/nanoalphazero/research/muzero/remat.py': '087a237e3ebe600df48261d70f832c17191c708117624e6387905196d5ac3b75',
                   'src/nanoalphazero/research/muzero/spatial.py': '5f382d581b4ca87f5b578732557ab1bdad10aeacc8fef0502fc098f2241617a1',
                   'src/nanoalphazero/research/muzero/search.py': '4d1d957cc3dbf75317a29d9861937c4cf9961e5344038209ce232703243214ad',
                   'src/nanoalphazero/research/muzero/replay.py': '203891929124dcd2f8755b6c5ed077e00a3dd5b5186851b63bcefbf6e16b350c',
                   'src/nanoalphazero/research/muzero/staging.py': '730a3a58ad2e7fe401ac667e776d5a5cb6b5cb66a1b1212fc218effebdec0ca2',
                   'src/nanoalphazero/research/muzero/learning.py': '23b740e9e0fb4b2fe5772ceb62d4a24163b7eaed2f709d5142a1ad31a47ede58',
                   'src/nanoalphazero/research/muzero/checkpoint.py': 'e0d3e8043e3b45fd55791b21eea17578af3c1e1ba8f7f6ccd89bafe8addb2ddf',
                   'src/nanoalphazero/research/muzero/charts.py': 'eec2ae5660831cee792268e1c76ce43013faec738259306a9a37741b6333ed0d',
                   'src/nanoalphazero/research/muzero/metrics.py': '0e6175267f629ee32cc22904409be4bbefc7763977ef058e4c563eab40bcf572',
                   'src/nanoalphazero/research/muzero/inspection.py': 'd3247fe4e13c289c02ac5a53f6e067ed3983906ce01b395c9859926a557bbfb5',
                   'src/nanoalphazero/research/muzero/decisions.py': 'e8dcbda0f0c16f211f8c4dca52c3c9f0399a168a904935fb8429634552096938',
                   'src/nanoalphazero/research/muzero/evaluation.py': 'cb91d9c5d8e2d61fe9641372176a56aa0ca59c8b3889c15bc9599d9b1d1dd656',
                   'src/nanoalphazero/research/muzero/convergence.py': 'c4cf21a6bebdc2297cd2366cda6719ea965cc008cd67249f1320491a44fd18da',
                   'src/nanoalphazero/research/muzero/cli.py': 'e5962d1ac15ab5de4d826e42fb6d0c630e4ef74235c07b360e808bf1bec88214'}}
STANDALONE_MOHEX_CONFIG = '# Strong, uncapped defaults matching the historical ~/az Hex validator.\nparam_mohex knowledge_threshold 0\nparam_mohex use_parallel_solver 1\nparam_dfpn threads 4\n\n'


def standalone_train_parser():
    parser = argparse.ArgumentParser(description="MuZero self-play training; no config file required.",
                                     epilog="Full running instructions: uv run muzero.py --help")
    parser.add_argument("config", type=Path, nargs="?", help="Optional advanced TOML overrides")
    parser.add_argument("--preset", choices=sorted(STANDALONE_PRESETS))
    parser.add_argument("--env", help="Game name (default: hex4)")
    parser.add_argument("--platform", choices=("cpu", "tpu"), help="Default: tpu")
    parser.add_argument("--network", choices=("vector", "spatial"))
    parser.add_argument("--output", type=Path, help="Default: unique directory under artifacts/")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--hex-eval-engine-path")
    parser.add_argument("--hex-eval-engine-config", default="default")
    parser.add_argument("--print-config", action="store_true", help="Print resolved settings and exit without training")
    parser.add_argument("--save", dest="save_checkpoints", action=argparse.BooleanOptionalAction, default=None)
    for name in ("wandb", "remat-unroll", "remat-blocks"):
        parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=None)
    for name in ("seed", "cycles", "width", "depth", "unroll", "train-batch-size", "selfplay-batch-size",
                 "checkpoint-period", "hex-eval-period", "eval-period"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--learning-rate", type=float)
    return parser


def standalone_train_settings(args):
    if args.preset or args.config:
        raw = dict(STANDALONE_PRESETS.get(args.preset, {}))
    else:
        raw = dict(env="hex4", defaults="alphazero", platform="tpu", devices=4,
                   network="spatial", data_pipeline="staged", wandb=True)
    if args.config:
        with args.config.open("rb") as stream:
            raw.update(tomllib.load(stream))
    for name in ("env", "platform", "network", "save_checkpoints", "wandb", "remat_unroll", "remat_blocks",
                 "seed", "cycles", "width", "depth", "unroll", "train_batch_size", "selfplay_batch_size",
                 "checkpoint_period", "hex_eval_period", "eval_period", "learning_rate"):
        value = getattr(args, name)
        if value is not None:
            raw[name] = value
    return raw


def standalone_output(config):
    from datetime import datetime, timezone
    import uuid
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("artifacts") / f"muzero-{config['env']}-{stamp}-{uuid.uuid4().hex[:8]}"


def standalone_bootstrap():
    """Select backend before third-party imports; never probe a running TPU."""
    if __name__ != "__main__":
        return
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("train", "eval") or any(a in ("-h", "--help") for a in argv):
        return
    if argv[0] == "train":
        args = standalone_train_parser().parse_args(argv[1:])
        raw = standalone_train_settings(args)
        if args.print_config:
            # Resolving config needs no accelerator, even when its target is TPU.
            os.environ["JAX_PLATFORMS"] = "cpu"
            return
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--platform", choices=("cpu", "tpu"), default="tpu")
        args, _ = parser.parse_known_args(argv[1:])
        raw = {"platform": args.platform}
    if "platform" in raw:
        os.environ["JAX_PLATFORMS"] = raw["platform"]
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        flag = "--xla_force_host_platform_device_count="
        if flag not in os.environ.get("XLA_FLAGS", ""):
            os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                       f" {flag}{raw.get('devices', 4)}").strip()


standalone_bootstrap()


def standalone_provenance():
    """Describe the exported source, even outside Git or inside another repo."""
    return {"source_snapshot": STANDALONE_SOURCE,
            "standalone_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def standalone_snapshot(output, manifest):
    import shutil
    source = Path(__file__)
    snapshot = output / "source-muzero"
    snapshot.mkdir()
    shutil.copyfile(source, snapshot / "muzero.py")
    manifest["source_sha256"] = {"muzero.py": hashlib.sha256(source.read_bytes()).hexdigest()}


def standalone_mohex_config():
    """Materialize only the optional engine's tiny config, never Python modules."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as stream:
        stream.write(STANDALONE_MOHEX_CONFIG)
        path = Path(stream.name)
    atexit.register(path.unlink, missing_ok=True)
    return path


def standalone_data_sharding():
    return jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(jax.devices()), ("x",)), jax.sharding.PartitionSpec("x"))


def standalone_main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    command = sys.argv.pop(1)
    if command == "train":
        training_main()
    elif command == "eval":
        evaluation_main()
    else:
        raise SystemExit(f"Unknown command {command!r}; use train or eval")


# Third-party libraries and shared standard-library imports.
import functools
from dataclasses import dataclass
from typing import Callable
from typing import Generic
from typing import Optional
import chex
import flashbax as fbx
import jax
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import Experience
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from jax import Array
from jax.typing import ArrayLike
import math
import re
from typing import Sequence
from typing import Tuple
import flax.linen as nn
import numpy as np
from typing import Any
import flax.traverse_util
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors_file
from dataclasses import field
from types import SimpleNamespace
from typing import NamedTuple
import optax
import pgx1
from flax.training import train_state
from pgx1.experimental import auto_reset
from concurrent.futures import ThreadPoolExecutor
import shutil
import subprocess
from typing import Protocol
import time
from datetime import datetime
from datetime import timezone
import flashbax
from flax import struct
from flax import serialization
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
from flax.traverse_util import empty_node
from safetensors.numpy import save_file


# =============================================================================
# Environment and AlphaZero-compatible training defaults
# Source: src/nanoalphazero/config.py
# =============================================================================

"""Built-in training configurations for supported games."""


# =============================================================================
# Configuration
# =============================================================================
def get_ttt_config():
    board_size = 3
    game_max_steps = board_size * board_size
    batch_size = 1024
    REPLAY_BUFFER_TOTAL_SIZE = 1_024_000

    selfplay_buffer_len = game_max_steps + 10
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": "tic_tac_toe",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": 4,
        "env_forbids_draws": False,
        "env_allows_draws": True,
        "boardsize": board_size,
        # game_obs_shape and game_num_actions are derived from the live env in make_alphazero
        "game_obs_shape": None,
        "game_num_actions": board_size * board_size,
        # --- Model ---
        "conv_width": 32,
        "conv_depth": 4,
        # --- MCTS ---
        "mcts_num_simulations": 13,
        "mcts_variant": "1sh",
        "mcts_max_m": 9,
        "mcts_num_root_considered": 9,
        "mcts_num_survivors": 4,
        "mcts_num_k_actions": 9,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,  # [todo] not implemented
        # --- Training ---
        "num_iters": 10_000,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "weight_decay_kernels_only": True,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": 10,
        "cycle_n_train": 10,
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": game_max_steps,
        "selfplay_buffer_max_len": game_max_steps,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 100,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "ckpt_period": None,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


def get_hex_config(board_size=4):
    board_cfgs = {
        4: dict(
            conv_width=64,
            conv_depth=4,
            num_iters=(500 * 10),
            learning_rate=1e-3,
            cycle_n_selfplay=10,
            cycle_n_train=10,
        ),
        5: dict(
            conv_width=128,
            conv_depth=4,
            num_iters=(1700 * 20),
            learning_rate=1e-4,
            cycle_n_selfplay=20,
            cycle_n_train=20,
        ),
        6: dict(
            conv_width=256,
            conv_depth=8,
            num_iters=(3500 * 30),
            learning_rate=1e-4,
            cycle_n_selfplay=30,
            cycle_n_train=30,
        ),
        7: dict(
            conv_width=256,
            conv_depth=16,
            num_iters=(5000 * 40),
            learning_rate=1e-4,
            cycle_n_selfplay=40,
            cycle_n_train=30,
        ),
        8: dict(
            conv_width=256,
            conv_depth=16,
            num_iters=(10_000 * 60),
            learning_rate=1e-4,
            cycle_n_selfplay=60,
            cycle_n_train=50,
        ),
        9: dict(
            conv_width=256,
            conv_depth=32,
            num_iters=(40_000 * 80),
            learning_rate=1e-4,
            cycle_n_selfplay=40,
            cycle_n_train=25,
        ),
    }
    if board_size not in board_cfgs:
        raise ValueError(f"Unsupported hex board size: {board_size}")
    board_cfg = board_cfgs[board_size]

    game_max_steps = board_size * board_size

    # Small solved boards need less parallelism; larger boards retain TPU-sized batches.
    batch_size = 1024 if board_size in (4, 5) else 8192
    REPLAY_BUFFER_TOTAL_SIZE = 2_048_000
    # ------------------------------------------------------------

    selfplay_buffer_len = game_max_steps + 20
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": f"hexnoswap_{board_size}x{board_size}",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": game_max_steps // 2,
        # hex has no draws: the last player to move always wins.
        "env_forbids_draws": True,
        "env_allows_draws": False,
        "boardsize": board_size,
        "game_obs_shape": None,
        "game_num_actions": None,  # patched from live env in make_alphazero
        # --- Model ---
        "conv_width": board_cfg["conv_width"],
        "conv_depth": board_cfg["conv_depth"],
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 24,
        "mcts_variant": "1sh",
        "mcts_max_m": 16,
        "mcts_num_root_considered": 16,
        "mcts_num_survivors": 8,
        "mcts_num_k_actions": game_max_steps,  # = board_size**2
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        # bnk off for hex (full (A,) policy targets); root temperature from prod hex
        "exp_bnk_action_weights": False,
        "exp_use_root_temperature": True,
        "exp_root_temperature": 1.3,
        # --- Training ---
        "num_iters": board_cfg["num_iters"],
        "learning_rate": board_cfg["learning_rate"],
        "weight_decay": 1e-4,
        "weight_decay_kernels_only": True,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": board_cfg["cycle_n_selfplay"],
        "cycle_n_train": board_cfg["cycle_n_train"],
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 50,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "eval_opening_plies": 2,
        "hex_eval_period": 0,
        "hex_eval_engine": "mohex",
        "hex_eval_engine_path": None,
        "hex_eval_engine_config": "default",
        "ckpt_period": None,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


def get_connect4_config():
    game_max_steps = 42  # 6 rows x 7 cols
    batch_size = 8192
    REPLAY_BUFFER_TOTAL_SIZE = 2_048_000

    selfplay_buffer_len = game_max_steps + 10
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": "connect_four",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": 21,
        # connect4 can end in a draw (full board), so draws are allowed.
        "env_forbids_draws": False,
        "env_allows_draws": True,
        "boardsize": 7,  # number of columns (= action space); board is 6x7
        "game_obs_shape": None,
        "game_num_actions": 7,
        # --- Model ---
        "conv_width": 128,
        "conv_depth": 8,
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 64,
        "mcts_variant": "1sh",
        "mcts_max_m": 7,
        "mcts_num_root_considered": 7,
        "mcts_num_survivors": 3,
        "mcts_num_k_actions": 7,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        "exp_bnk_action_weights": False,
        # --- Training ---
        "num_iters": 2100 * 20,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "weight_decay_kernels_only": True,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": 20,
        "cycle_n_train": 12,
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 50,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "eval_opening_plies": 3,
        "ckpt_period": 700,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


def get_chess_config():
    board_size = 8
    GAME_MAX_STEPS = 512

    selfplay_bs = 8192
    train_bs = 8192
    REPLAY_BUFFER_TOTAL_SIZE = 4096000 * 2
    # -----------------------------------------------------------------------

    selfplay_buffer_len = GAME_MAX_STEPS + 20
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // train_bs
    buffer_warmup_steps = selfplay_buffer_len + replay_buffer_len

    return {
        # --- Game ---
        "env_id": "chess",
        "game_max_steps": GAME_MAX_STEPS,
        "num_exploratory_moves": 30,
        "env_forbids_draws": False,
        "env_allows_draws": True,
        "boardsize": board_size,
        "game_obs_shape": None,
        "game_num_actions": None,  # patched from live env in make_alphazero
        # --- Model ---
        "conv_width": 256,
        "conv_depth": 10,
        "katago_preset": "b10c256nbt",
        "katago_activation": "mish",
        "katago_use_rvgl": True,
        "use_wdl": True,
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 12,
        "mcts_variant": "1sh",
        "mcts_max_m": 8,
        "mcts_num_root_considered": 8,
        "mcts_num_survivors": 4,
        "mcts_num_k_actions": 128,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        # bnk: store compressed (K,) policy targets instead of full (4672,)
        "exp_bnk_action_weights": True,
        "exp_use_root_temperature": True,  # from KataGo
        "exp_root_temperature": 1.5,
        # --- Training ---
        "num_iters": 459_000 * 20,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "weight_decay_kernels_only": True,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": train_bs,
        "cycle_n_selfplay": 20,
        "cycle_n_train": 20,
        # --- Self-play & Buffers ---
        "selfplay_batch_size": selfplay_bs,
        "selfplay_buffer_add_batch_size": selfplay_bs,
        "selfplay_buffer_sample_batch_size": train_bs,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": train_bs,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": train_bs,
        "replay_buffer_sample_batch_size": train_bs,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        "diagnostic_period": 20,
        # --- Strength eval (vs random + frozen anchor; no external engine) ---
        # Plays one game per legal opening (chess = 20) from both colors.
        "eval_period": 50,  # run every N cycles
        # Forced opening depth per eval game: enumerate every legal line of this
        # many plies (1 -> 20 chess openings, 2 -> ~400) and play one game each
        # from both colors. Deeper = wider opening coverage = lower-variance,
        # more accurate ladder score, at proportionally more eval games.
        "eval_opening_plies": 2,
        "eval_max_plies": 200,  # cap match length; unfinished games = draw
        # Max Elo credited for beating random (rung 0's anchor). At 0, beating
        # random just anchors the ladder at Elo 0 and all real Elo comes from
        # beating past selves. Without a cap, ~100% vs random pegs ΔElo at the
        # clamp ceiling (≈+1600) and inflates the whole ladder. Set None to
        # disable the cap entirely.
        "eval_vs_random_max_elo": 0.0,
        "ckpt_period": 800,  # save a checkpoint every 800 cycles
        # --- System ---
        "enable_sharding": True,
        "debug_probe_executables": True,
    }


def get_go_config(board_size=5):
    # pgx1 Go defaults komi to a per-size lookup (all X.5 values, 7.5
    # fallback), so games never draw (score margin is always non-integer):
    # the env always resolves to a win for one side. We treat it like hex
    # (env_forbids_draws=True). The action space is board_size**2 board
    # points + 1 pass, and a game can run up to board_size**2 * 2 plies
    # (pgx1's default max_terminal_steps), so the buffers are sized for that.
    board_cfgs = {
        3: dict(
            conv_width=64,
            conv_depth=4,
            num_iters=(1500 * 10),
            learning_rate=1e-3,
            cycle_n_selfplay=10,
            cycle_n_train=10,
            num_root_considered=8,
            num_survivors=4,
        ),
        4: dict(
            conv_width=128,
            conv_depth=6,
            num_iters=(2000 * 20),
            learning_rate=1e-4,
            cycle_n_selfplay=20,
            cycle_n_train=20,
            num_root_considered=16,
            num_survivors=8,
        ),
        5: dict(
            conv_width=256,
            conv_depth=8,
            num_iters=(5000 * 30),
            learning_rate=1e-4,
            cycle_n_selfplay=30,
            cycle_n_train=25,
            num_root_considered=16,
            num_survivors=8,
        ),
        6: dict(
            conv_width=256,
            conv_depth=8,
            num_iters=(10000 * 40),
            learning_rate=1e-4,
            cycle_n_selfplay=40,
            cycle_n_train=40,
            num_root_considered=8,
            num_survivors=4,
        ),
        7: dict(
            conv_width=256,
            conv_depth=16,
            num_iters=(20000 * 50),
            learning_rate=1e-4,
            cycle_n_selfplay=50,
            cycle_n_train=50,
            num_root_considered=8,
            num_survivors=4,
        ),
        8: dict(
            conv_width=256,
            conv_depth=32,
            num_iters=(50000 * 50),
            learning_rate=1e-4,
            cycle_n_selfplay=50,
            cycle_n_train=50,
            num_root_considered=8,
            num_survivors=4,
        ),
        9: dict(
            conv_width=256,
            conv_depth=16,
            num_iters=(200000 * 50),
            learning_rate=1e-4,
            cycle_n_selfplay=50,
            cycle_n_train=50,
            num_root_considered=8,
            num_survivors=4,
        ),
    }
    if board_size not in board_cfgs:
        raise ValueError(f"Unsupported go board size: {board_size}")
    board_cfg = board_cfgs[board_size]

    # board points + pass; a game can last up to size**2 * 2 plies.
    num_actions = board_size * board_size + 1
    game_max_steps = board_size * board_size * 2

    batch_size = 8192
    REPLAY_BUFFER_TOTAL_SIZE = 2_048_000

    selfplay_buffer_len = game_max_steps + 20
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": f"go_{board_size}x{board_size}",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": game_max_steps // 2,
        # komi 7.5 => no draws (one side always wins on score).
        "env_forbids_draws": True,
        "env_allows_draws": False,
        "boardsize": board_size,
        "game_obs_shape": None,
        "game_num_actions": None,  # patched from live env in make_alphazero
        # --- Model ---
        "conv_width": board_cfg["conv_width"],
        "conv_depth": board_cfg["conv_depth"],
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 24,
        "mcts_variant": "1sh",
        "mcts_max_m": board_cfg["num_root_considered"],
        "mcts_num_root_considered": board_cfg["num_root_considered"],
        "mcts_num_survivors": board_cfg["num_survivors"],
        "mcts_num_k_actions": num_actions,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        "exp_bnk_action_weights": False,
        "exp_use_root_temperature": True,
        "exp_root_temperature": 1.3,
        # --- Training ---
        "num_iters": board_cfg["num_iters"],
        "learning_rate": board_cfg["learning_rate"],
        "weight_decay": 1e-4,
        "weight_decay_kernels_only": True,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": board_cfg["cycle_n_selfplay"],
        "cycle_n_train": board_cfg["cycle_n_train"],
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 50,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "eval_opening_plies": 2,
        "ckpt_period": None,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


CONFIG_FACTORIES = {
    "chess": get_chess_config,
    "ttt": get_ttt_config,
    "connect4": get_connect4_config,
    "hex4": lambda: get_hex_config(board_size=4),
    "hex5": lambda: get_hex_config(board_size=5),
    "hex6": lambda: get_hex_config(board_size=6),
    "hex7": lambda: get_hex_config(board_size=7),
    "hex8": lambda: get_hex_config(board_size=8),
    "hex9": lambda: get_hex_config(board_size=9),
    "go3": lambda: get_go_config(board_size=3),
    "go4": lambda: get_go_config(board_size=4),
    "go5": lambda: get_go_config(board_size=5),
    "go6": lambda: get_go_config(board_size=6),
    "go7": lambda: get_go_config(board_size=7),
    "go8": lambda: get_go_config(board_size=8),
    "go9": lambda: get_go_config(board_size=9),
}


# =============================================================================
# Shared buffer primitives and packed chess legality
# Source: src/nanoalphazero/buffers.py
# =============================================================================

"""Replay/self-play buffers and their stored sample representation."""


# =============================================================================
# Self-play records
# =============================================================================
@chex.dataclass(frozen=True)
class SelfplayOutput:
    col_id: ArrayLike
    row_id: ArrayLike
    global_step_id: Optional[ArrayLike]
    game_id: Optional[ArrayLike]
    action: ArrayLike
    action_weights: ArrayLike
    reward: ArrayLike
    is_valid_sample: ArrayLike
    is_from_selfplay: ArrayLike
    player: Optional[ArrayLike]
    just_terminated: Optional[ArrayLike]
    ep_step: Optional[ArrayLike]
    ep_termination_step: Optional[ArrayLike]
    is_exploration: Optional[ArrayLike]
    is_pending_reward_i8: Optional[ArrayLike]
    is_fresh_i8: Optional[ArrayLike]
    k_indices: Optional[ArrayLike] = None
    observation: Optional[ArrayLike] = None
    legal_action_mask: Optional[ArrayLike] = None
    # compressed chess fields
    board_bool: Optional[ArrayLike] = None
    board_float: Optional[ArrayLike] = None
    legal_action_bitmask: Optional[ArrayLike] = None


def split_observation(obs_array):
    bool_indices = jnp.concatenate([jnp.arange(113), jnp.arange(114, 118)])
    float_indices = jnp.array([113, 118])
    bool_part = obs_array[:, :, bool_indices].astype(jnp.bool_)
    float_part = obs_array[:, :, float_indices].astype(jnp.bfloat16)
    packed_bool_part = jnp.packbits(bool_part.flatten())
    return packed_bool_part, float_part


split_observation_vmap = jax.vmap(split_observation)


def combine_observation(packed_bool_part, float_part):
    bool_flat = jnp.unpackbits(packed_bool_part)
    bool_part = bool_flat.reshape((8, 8, 117))
    obs_reconstructed = jnp.zeros((8, 8, 119), dtype=jnp.float32)
    bool_indices = jnp.concatenate([jnp.arange(113), jnp.arange(114, 118)])
    obs_reconstructed = obs_reconstructed.at[:, :, bool_indices].set(
        bool_part.astype(jnp.float32)
    )
    float_indices = jnp.array([113, 118])
    obs_reconstructed = obs_reconstructed.at[:, :, float_indices].set(
        float_part.astype(jnp.float32)
    )
    return obs_reconstructed


combine_observation_vmap = jax.vmap(combine_observation)

NUM_ACTIONS = 4672
NUM_WORDS = (NUM_ACTIONS + 31) // 32  # 146


def pack_mask(mask):
    reshaped_mask = mask.reshape(NUM_WORDS, 32)
    powers_of_2 = jnp.left_shift(jnp.uint32(1), jnp.arange(32, dtype=jnp.uint32))
    return jnp.sum(reshaped_mask * powers_of_2, axis=1, dtype=jnp.uint32)


def unpack_bitmask(bitset):
    powers_of_2 = jnp.left_shift(jnp.uint32(1), jnp.arange(32, dtype=jnp.uint32))
    return ((bitset[:, None] & powers_of_2[None, :]) > 0).flatten()


pack_mask_vmap = jax.vmap(pack_mask)
unpack_bitmask_vmap = jax.vmap(unpack_bitmask)

# =============================================================================
# Replay & self-play buffers
# =============================================================================
@chex.dataclass(frozen=True)
class CustomTrajectoryBufferState(TrajectoryBufferState[Experience]):
    num_valid_consumable: jax.Array = 0


@dataclass(frozen=True)
class Buffer(TrajectoryBuffer, Generic[Experience]):
    add_backfill: Optional[
        Callable[
            [CustomTrajectoryBufferState[Experience], Experience, Array, Array],
            tuple[CustomTrajectoryBufferState[Experience], dict],
        ]
    ] = None
    consume: Optional[
        Callable[
            [CustomTrajectoryBufferState[Experience]],
            tuple[CustomTrajectoryBufferState[Experience], Experience, dict],
        ]
    ] = None


def get_dummy_selfplay_output(config) -> SelfplayOutput:
    num_actions = config["game_num_actions"]
    obs_shape = config["game_obs_shape"]  # patched by make_alphazero from the live env
    is_chess = config["env_id"] == "chess"
    # build a kwargs dict so chess/bnk variants can override fields
    common = dict(
        col_id=jnp.zeros([], dtype=jnp.uint32),
        row_id=jnp.zeros([], dtype=jnp.uint32),
        global_step_id=jnp.zeros([], dtype=jnp.uint32),
        game_id=jnp.zeros([], dtype=jnp.uint32),
        action=jnp.zeros([], dtype=jnp.int32),
        action_weights=jnp.zeros((num_actions,), dtype=jnp.float32),
        reward=jnp.zeros([], dtype=jnp.float32),
        is_from_selfplay=jnp.zeros([], dtype=jnp.bool_),
        player=jnp.full([], -1, dtype=jnp.int32),
        just_terminated=jnp.zeros([], dtype=jnp.bool_),
        ep_step=jnp.full([], -127, dtype=jnp.int16),
        ep_termination_step=jnp.full([], 0, dtype=jnp.int16),
        is_exploration=jnp.zeros([], dtype=jnp.bool_),
        is_pending_reward_i8=jnp.ones([], dtype=jnp.int8),
        is_fresh_i8=jnp.zeros([], dtype=jnp.int8),
        is_valid_sample=jnp.zeros([], dtype=jnp.bool_),
    )
    # bnk stores (K,) policy targets + k_indices instead of full (A,)
    if config.get("exp_bnk_action_weights", False):
        k = config["mcts_num_k_actions"]
        common["action_weights"] = jnp.zeros((k,), dtype=jnp.float32)
        common["k_indices"] = jnp.zeros((k,), dtype=jnp.int32)
    # chess stores compressed obs + bitmask; others store raw obs/mask
    if is_chess:
        common["board_bool"] = jnp.zeros((936,), dtype=jnp.uint8)
        common["board_float"] = jnp.zeros((8, 8, 2), dtype=jnp.bfloat16)
        common["legal_action_bitmask"] = jnp.zeros((NUM_WORDS,), dtype=jnp.uint32)
    else:
        common["observation"] = jnp.zeros(obs_shape, dtype=jnp.bool_)
        common["legal_action_mask"] = jnp.zeros((num_actions,), dtype=jnp.bool_)
    return SelfplayOutput(**common)


def make_replay_buffer(config, dummy_selfplay_output, data_sharding=None):
    # The replay buffer holds finished training samples: positions whose final
    # reward has already been filled in (backfilled from the game's outcome).
    # Phase 3 samples gradient batches from here. Nothing in this buffer is
    # "in progress" -- by the time a sample lands here it is complete and
    # trainable. It is fed by draining the selfplay buffer (see below).
    replay_buffer = fbx.make_trajectory_buffer(
        add_batch_size=config["replay_buffer_add_batch_size"],
        sample_batch_size=config["replay_buffer_sample_batch_size"],
        sample_sequence_length=1,
        period=1,
        min_length_time_axis=config["replay_buffer_min_len"],
        max_length_time_axis=config["replay_buffer_max_len"],
    )
    replay_buffer = replay_buffer.replace(
        add=jax.jit(replay_buffer.add, donate_argnums=0),
        can_sample=jax.jit(replay_buffer.can_sample),
    )

    # data-parallel sharded init + sample, replicated for scalars
    if config.get("enable_sharding", False) and data_sharding is not None:
        replicated_sharding = jax.sharding.NamedSharding(
            data_sharding.mesh, jax.sharding.PartitionSpec()
        )
        sample_fn = jax.jit(replay_buffer.sample, out_shardings=data_sharding)
        state_shape_tree = jax.eval_shape(replay_buffer.init, dummy_selfplay_output)

        def _spec(shape_struct):
            return data_sharding if shape_struct.ndim > 0 else replicated_sharding

        out_sharding_tree = jax.tree_util.tree_map(_spec, state_shape_tree)
        init_fn = jax.jit(replay_buffer.init, out_shardings=out_sharding_tree)
        with data_sharding.mesh:
            replay_buffer_state = init_fn(dummy_selfplay_output)
    else:
        sample_fn = jax.jit(replay_buffer.sample)
        init_fn = jax.jit(replay_buffer.init)
        replay_buffer_state = init_fn(dummy_selfplay_output)

    buffer = Buffer(
        init=init_fn,
        add=replay_buffer.add,
        sample=sample_fn,
        can_sample=replay_buffer.can_sample,
    )
    return buffer, replay_buffer_state


def make_selfplay_buffer(config, dummy_selfplay_output, data_sharding=None):
    # The selfplay buffer is a staging area for games that are still in progress.
    # Positions are written here as games are played, but they don't yet have a
    # reward -- `add_backfill` fills the reward in once the game terminates. Once
    # a position has its reward it is "consumed": handed off to the replay buffer
    # and marked is_fresh=False.
    #
    # The is_fresh flag is the key to correctness. is_fresh=True means "this
    # position just received its reward and has not been consumed yet". `consume`
    # only ever returns fresh positions and immediately flips them to
    # is_fresh=False. Without this, the same positions could be returned over and
    # over due to how top_k works.
    selfplay_buffer = fbx.make_trajectory_buffer(
        add_batch_size=config["selfplay_buffer_add_batch_size"],
        sample_batch_size=config["selfplay_buffer_sample_batch_size"],
        sample_sequence_length=1,
        period=1,
        min_length_time_axis=config["selfplay_buffer_min_len"],
        max_length_time_axis=config["selfplay_buffer_max_len"],
    )
    selfplay_buffer = selfplay_buffer.replace(
        add=jax.jit(selfplay_buffer.add, donate_argnums=0),
        sample=jax.jit(selfplay_buffer.sample),
        can_sample=jax.jit(selfplay_buffer.can_sample),
    )

    # sharded init for selfplay buffer state, REPLICATED scalar counter
    if config.get("enable_sharding", False) and data_sharding is not None:
        replicated_sharding = jax.sharding.NamedSharding(
            data_sharding.mesh, jax.sharding.PartitionSpec()
        )
        state_shape_tree = jax.eval_shape(selfplay_buffer.init, dummy_selfplay_output)

        def _spec(shape_struct):
            return data_sharding if shape_struct.ndim > 0 else replicated_sharding

        out_sharding_tree = jax.tree_util.tree_map(_spec, state_shape_tree)
        init_fn = jax.jit(selfplay_buffer.init, out_shardings=out_sharding_tree)
        with data_sharding.mesh:
            selfplay_buffer_state = init_fn(dummy_selfplay_output)
            selfplay_buffer_state = CustomTrajectoryBufferState(
                experience=selfplay_buffer_state.experience,
                current_index=selfplay_buffer_state.current_index,
                is_full=selfplay_buffer_state.is_full,
                num_valid_consumable=jax.lax.with_sharding_constraint(
                    jnp.array(0, dtype=jnp.int32), replicated_sharding
                ),
            )
    else:
        init_fn = jax.jit(selfplay_buffer.init)
        selfplay_buffer_state = init_fn(dummy_selfplay_output)
        selfplay_buffer_state = CustomTrajectoryBufferState(
            experience=selfplay_buffer_state.experience,
            current_index=selfplay_buffer_state.current_index,
            is_full=selfplay_buffer_state.is_full,
            num_valid_consumable=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, donate_argnums=(0,))
    def add_backfill(
        selfplay_buffer_state,
        selfplay_output,
        env_state_terminated,
        env_state_rewards,
    ):
        """Append a fresh slice of selfplay data and backfill rewards onto past positions.

        This is one of the trickiest parts of the RL cycle, and the cause of countless
        headaches. Read it carefully before touching anything.

        The core problem: in self-play we generate the positions of a game *before* we
        know who won. A position is written into the buffer at the step it was played,
        but its reward (+1/-1/0) only becomes known later, at the moment the game
        terminates. So `add()` writes positions with a placeholder reward of 0, and on
        every subsequent call we have to find the positions belonging to games that have
        *just* terminated and patch their rewards in-place. That patching is what
        "backfill" means here.

        Several bookkeeping flags coordinate this:

          - `is_from_selfplay`  : the slot holds real self-play data (vs. uninitialized
                                  default data that's just sitting in the buffer).
          - `is_pending_reward_i8`: this position is still waiting for its terminal
                                  reward to be filled in.
          - `is_fresh_i8`       : this position just received its reward and hasn't been
                                  handed downstream yet.
          - `is_valid_sample`   : this position is fully formed and eligible to be
                                  consumed into the replay buffer and, subsequently,
                                  valid for training.

        The flow below is, roughly: (1) identify which buffer slots belong to a game
        that terminated this step, (2) add the now-known per-player reward onto those
        slots, (3) flip their pending/fresh/valid flags accordingly, and (4) propagate
        a bit of per-game metadata (termination step, game id) used only for logging.
        """
        selfplay_buffer_state = selfplay_buffer.add(
            selfplay_buffer_state, selfplay_output
        )

        # `is_from_selfplay` is necessary to make sure we're not operating on default
        # data sitting in the selfplay buffer. We only want to touch data that actually
        # came out of selfplay_fn. `real_samples` narrows that further to slots that are
        # still awaiting their terminal reward (is_pending_reward_i8).
        is_from_selfplay = selfplay_buffer_state.experience.is_from_selfplay
        real_samples = (
            is_from_selfplay & selfplay_buffer_state.experience.is_pending_reward_i8
        )

        # Of those pending slots, the ones whose game terminated *this* step are the ones
        # we now have a reward for. Split them by which player the position belongs to so
        # we can assign each side its own +1/-1.
        entries_to_update_mask = real_samples * env_state_terminated[:, None]
        player1_entries = entries_to_update_mask * (
            selfplay_buffer_state.experience.player == 0
        )
        player2_entries = entries_to_update_mask * (
            selfplay_buffer_state.experience.player == 1
        )

        player1_rewards = env_state_rewards[:, 0].reshape(-1, 1)
        player2_rewards = env_state_rewards[:, 1].reshape(-1, 1)

        # The reward-backfill trick. SelfplayOutput is always emitted with reward == 0
        # (see selfplay_fn), so adding the terminal reward onto the existing value is
        # equivalent to a masked write: untouched slots keep their 0 (or prior reward),
        # and only the just-terminated slots for each player pick up their +1/-1.
        old_experience = selfplay_buffer_state.experience
        new_rewards = old_experience.reward + (player1_entries * player1_rewards)
        new_rewards = new_rewards + (player2_entries * player2_rewards)
        # A slot that just got its reward is no longer pending, and is now "fresh":
        # carrying a brand-new reward that downstream hasn't seen yet. `consume` clears
        # is_fresh_i8 the first time it hands a slot out, so we never systematically
        # return the same data over and over — that repetition would inject a subtle
        # sampling bias into training.
        new_is_pending_reward_i8 = (
            old_experience.is_pending_reward_i8 - entries_to_update_mask
        )
        new_is_fresh_i8 = old_experience.is_fresh_i8 + entries_to_update_mask

        # A slot becomes a valid, consumable sample once it is real self-play data, not
        # an exploration move, freshly rewarded, and no longer pending. For games that
        # forbid draws we additionally require a non-zero reward as a guard: a 0 reward
        # there can only mean the backfill hasn't actually landed yet.
        assert config["env_allows_draws"] != config["env_forbids_draws"]
        if config["env_allows_draws"]:
            new_is_valid_sample = (
                is_from_selfplay
                & (~old_experience.is_exploration)
                & (new_is_fresh_i8 == 1)
                & (new_is_pending_reward_i8 == 0)
            )
        else:
            new_is_valid_sample = (
                is_from_selfplay
                & (~old_experience.is_exploration)
                & (new_rewards != 0)
                & (new_is_fresh_i8 == 1)
                & (new_is_pending_reward_i8 == 0)
            )

        # Propagate per-game metadata onto the just-terminated slots. This is purely for
        # logging/metrics (e.g. game length and per-game grouping) — it does not affect
        # training. Same masked-write pattern: where(entries_to_update_mask, new, old).
        newly_added_term_step = selfplay_output.ep_termination_step
        old_term_steps = old_experience.ep_termination_step
        update_values = jnp.broadcast_to(newly_added_term_step, old_term_steps.shape)
        new_ep_termination_step = jnp.where(
            entries_to_update_mask, update_values, old_term_steps
        )

        old_game_ids = old_experience.game_id
        update_game_ids = jnp.broadcast_to(selfplay_output.game_id, old_game_ids.shape)
        new_game_id = jnp.where(entries_to_update_mask, update_game_ids, old_game_ids)

        new_experience = selfplay_buffer_state.experience.replace(
            reward=new_rewards,
            is_pending_reward_i8=new_is_pending_reward_i8,
            is_fresh_i8=new_is_fresh_i8,
            is_valid_sample=new_is_valid_sample,
            ep_termination_step=new_ep_termination_step,
            game_id=new_game_id,
        )
        selfplay_buffer_state = selfplay_buffer_state.replace(
            experience=new_experience,
            num_valid_consumable=jnp.sum(new_is_valid_sample),
        )

        return selfplay_buffer_state, ({}, {})

    @functools.partial(jax.jit, donate_argnums=(0,))
    def consume(selfplay_buffer_state):
        new_is_fresh_i8 = selfplay_buffer_state.experience.is_fresh_i8
        new_is_valid_sample = selfplay_buffer_state.experience.is_valid_sample

        k = config["selfplay_buffer_consume_size"]
        B, T = new_is_fresh_i8.shape

        returnable_mask = new_is_valid_sample
        returnable_mask_flat = returnable_mask.flatten()
        is_fresh_i8_flat = new_is_fresh_i8.flatten()

        consume_seed = jnp.max(selfplay_buffer_state.experience.global_step_id).astype(
            jnp.uint32
        )
        consume_rng = jax.random.key(consume_seed)
        noise = jax.random.uniform(consume_rng, shape=returnable_mask_flat.shape)
        scores = jnp.where(returnable_mask_flat, noise, -jnp.inf)
        _, top_indices = jax.lax.top_k(scores, k=k)

        experience_flat = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]), selfplay_buffer_state.experience
        )
        completed_states = jax.tree.map(lambda x: x[top_indices], experience_flat)

        valid_selection = returnable_mask_flat[top_indices]
        completed_states = jax.tree.map(
            lambda x: jnp.where(valid_selection, x, jnp.zeros_like(x))
            if x.ndim <= 1
            else jnp.where(
                jnp.expand_dims(valid_selection, axis=tuple(range(1, x.ndim))),
                x,
                jnp.zeros_like(x),
            ),
            completed_states,
        )

        completed_games_with_time_axis = jax.tree.map(
            lambda x: jnp.expand_dims(x, axis=1), completed_states
        )
        is_fresh_i8_flat_after_update = is_fresh_i8_flat.at[top_indices].set(
            jnp.int8(0)
        )
        is_fresh_i8_after_return = is_fresh_i8_flat_after_update.reshape(B, T)

        new_is_valid_after_consume = (
            selfplay_buffer_state.experience.is_valid_sample
            & (is_fresh_i8_after_return == 1)
        )
        new_experience = selfplay_buffer_state.experience.replace(
            is_fresh_i8=is_fresh_i8_after_return,
            is_valid_sample=new_is_valid_after_consume,
        )
        selfplay_buffer_state = selfplay_buffer_state.replace(
            experience=new_experience,
            num_valid_consumable=jnp.sum(new_is_valid_after_consume),
        )

        return selfplay_buffer_state, completed_games_with_time_axis, {}

    buffer = Buffer(
        init=init_fn,
        add=selfplay_buffer.add,
        add_backfill=add_backfill,
        consume=consume,
        sample=selfplay_buffer.sample,
        can_sample=selfplay_buffer.can_sample,
    )

    return buffer, selfplay_buffer_state


# =============================================================================
# Shared KataGo network blocks and AlphaZero reference model
# Source: src/nanoalphazero/model.py
# =============================================================================

"""KataGo-style neural-network model definitions."""


# =============================================================================
# Neural network model
#
# KataGo's fixup nested-bottleneck architecture is the sole network used by
# nanoAlphaZero. The trunk is game-agnostic. Go uses KataGo's native board-point
# plus pass policy head; other games use a generic action-space policy head.
# MCTS consumes the scalar P(win)-P(loss), while training uses the raw WDL logits.
# =============================================================================
_TRUNC_STD_CORRECTION = 0.87962566103423978

_GAINS = {
    "relu": math.sqrt(2.0),
    "elu": math.sqrt(1.55052),
    "mish": math.sqrt(2.210277),
    "silu": math.sqrt(2.0),
    "gelu": math.sqrt(2.351718),
    "hardswish": math.sqrt(2.0),
    "identity": 1.0,
}


def _mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


_ACTS = {
    "relu": jax.nn.relu,
    "elu": jax.nn.elu,
    "mish": _mish,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "hardswish": jax.nn.hard_swish,
    "identity": lambda x: x,
}


def kata_init(scale: float, activation: str, fan_in: Optional[int] = None):
    """KataGo's ``init_weights``: trunc normal, std = scale*gain/sqrt(fan_in).

    ``fan_in`` defaults to prod(shape[:-1]), which matches torch's fan-in for
    both HWIO conv kernels (kh*kw*c_in) and (in, out) dense kernels. Pass it
    explicitly for biases (KataGo's ``fan_tensor=weight``).
    """
    gain = _GAINS[activation]

    def init(key, shape, dtype=jnp.float32):
        fi = fan_in if fan_in is not None else int(np.prod(shape[:-1]))
        std = scale * gain / math.sqrt(fi) / _TRUNC_STD_CORRECTION
        if std < 1e-10:
            return jnp.zeros(shape, dtype)
        return std * jax.random.truncated_normal(key, -2.0, 2.0, shape, dtype)

    return init


class NormMask(nn.Module):
    """KataGo's NormMask at norm_kind="fixup": ``(x*(1+gamma) + beta) * mask``.

    ``use_gamma`` corresponds to fixup_use_gamma=True (the second NormActConv
    of each ResBlock and the closing 1x1 of each nested block). All base
    configs set gamma_weight_decay_center_1, so gamma is stored centered at 0
    and applied as (gamma + 1). Also covers BiasMask (use_gamma=False).
    """

    use_gamma: bool = False

    @nn.compact
    def __call__(self, x, mask):
        c = x.shape[-1]
        beta = self.param("beta", nn.initializers.zeros, (c,))
        if self.use_gamma:
            gamma = self.param("gamma", nn.initializers.zeros, (c,))
            x = x * (gamma + 1.0)
        return (x + beta) * mask


def kata_gpool(x, mask, mask_sum_hw):
    """KataGPool: masked mean, size-scaled mean, masked max. [N,H,W,C] -> [N,3C].

    Assumes off-board activations are exactly 0 (guaranteed by NormMask) and
    that the activation maps 0 -> 0 and is > -1, so ``x + (mask - 1)`` makes
    off-board positions lose the max.
    """
    layer_mean = jnp.sum(x, axis=(1, 2)) / mask_sum_hw  # [N, C]
    s = jnp.sqrt(mask_sum_hw) - 14.0  # [N, 1]
    layer_max = jnp.max(x + (mask - 1.0), axis=(1, 2))  # [N, C]
    return jnp.concatenate([layer_mean, layer_mean * (s / 10.0), layer_max], axis=1)


def kata_value_head_gpool(x, mask, mask_sum_hw):
    """KataValueHeadGPool: mean with linear and quadratic board-size features."""
    layer_mean = jnp.sum(x, axis=(1, 2)) / mask_sum_hw
    s = jnp.sqrt(mask_sum_hw) - 14.0
    return jnp.concatenate(
        [
            layer_mean,
            layer_mean * (s / 10.0),
            layer_mean * (jnp.square(s) / 100.0 - 0.1),
        ],
        axis=1,
    )


class KataConvAndGPool(nn.Module):
    """Regular 3x3 conv branch + gpool branch feeding a bias back in."""

    c_out: int
    c_gpool: int
    scale: float  # fixup init scale for this position in the net
    activation: str

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        act = _ACTS[self.activation]
        # Fixup branch of KataConvAndGPool.initialize: r_scale=0.8 on the
        # regular conv, sqrt(scale)*sqrt(0.6) on both halves of the g branch.
        outr = nn.Conv(
            self.c_out, (3, 3), use_bias=False,
            kernel_init=kata_init(self.scale * 0.8, self.activation),
        )(x)
        g_scale = math.sqrt(self.scale) * math.sqrt(0.6)
        outg = nn.Conv(
            self.c_gpool, (3, 3), use_bias=False,
            kernel_init=kata_init(g_scale, self.activation),
        )(x)
        outg = NormMask()(outg, mask)
        outg = act(outg)
        outg = kata_gpool(outg, mask, mask_sum_hw)  # [N, 3*c_gpool]
        outg = nn.Dense(
            self.c_out, use_bias=False,
            kernel_init=kata_init(g_scale, self.activation),
        )(outg)
        return outr + outg[:, None, None, :]


class RepVGGLinearConv(nn.Module):
    """RVGL's separate 3x3/1x1 parameters evaluated as one fused convolution."""

    c_out: int
    kernel_size: int
    scale: float
    activation: str

    @nn.compact
    def __call__(self, x):
        c_in = x.shape[-1]
        kernel3 = self.param(
            "kernel_3x3",
            kata_init(self.scale * 0.8, self.activation),
            (self.kernel_size, self.kernel_size, c_in, self.c_out),
        )
        kernel1 = self.param(
            "kernel_1x1",
            kata_init(self.scale * 0.6, self.activation),
            (1, 1, c_in, self.c_out),
        )
        center = self.kernel_size // 2
        combined_kernel = kernel3.at[center, center].add(kernel1[0, 0])
        combined_kernel = combined_kernel.astype(x.dtype)
        return jax.lax.conv_general_dilated(
            lhs=x,
            rhs=combined_kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )


class NormActConv(nn.Module):
    """norm -> act -> conv (or conv+gpool)."""

    c_out: int
    kernel_size: int
    scale: float
    activation: str
    c_gpool: Optional[int] = None
    use_gamma: bool = False  # fixup_use_gamma
    use_rvgl: bool = True

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        out = NormMask(use_gamma=self.use_gamma)(x, mask)
        out = _ACTS[self.activation](out)
        if self.c_gpool is not None:
            return KataConvAndGPool(
                self.c_out, self.c_gpool, self.scale, self.activation
            )(out, mask, mask_sum_hw)
        if self.use_rvgl and self.kernel_size > 1:
            # KataGo -rvgl: parallel linear 3x3 and 1x1 branches. Their
            # initialization variances add to one (0.8**2 + 0.6**2 == 1).
            # Since there is no nonlinearity between the branches, the 1x1
            # kernel is folded into the 3x3 for the actual convolution while
            # remaining a separate parameter with its own optimizer state.
            return RepVGGLinearConv(
                self.c_out, self.kernel_size, self.scale, self.activation
            )(out)
        return nn.Conv(
            self.c_out, (self.kernel_size, self.kernel_size), use_bias=False,
            kernel_init=kata_init(self.scale, self.activation),
        )(out)


class ResBlock(nn.Module):
    """Inner two-conv residual block. Returns the residual only."""

    c_main: int
    c_mid: int
    fixup_scale: float
    activation: str
    c_gpool: Optional[int] = None
    use_rvgl: bool = True

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        c1_out = self.c_mid - (self.c_gpool or 0)
        out = NormActConv(
            c1_out, 3, self.fixup_scale, self.activation,
            c_gpool=self.c_gpool, use_rvgl=self.use_rvgl,
        )(x, mask, mask_sum_hw)
        # Fixup: second conv zero-init, and its NormMask carries a gamma.
        out = NormActConv(
            self.c_main, 3, 0.0, self.activation,
            use_gamma=True, use_rvgl=self.use_rvgl,
        )(out, mask, mask_sum_hw)
        return out


class NestedBottleneckResBlock(nn.Module):
    """KataGo "bottlenest{L}" block: 1x1 down, L residual ResBlocks, 1x1 up.

    Returns the residual only; the caller adds it to the trunk.
    """

    c_trunk: int
    c_mid: int
    internal_length: int
    fixup_scale: float
    activation: str
    c_gpool: Optional[int] = None
    use_rvgl: bool = True

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        inner_scale = self.fixup_scale ** (1.0 / (1.0 + self.internal_length))
        out = NormActConv(self.c_mid, 1, inner_scale, self.activation)(
            x, mask, mask_sum_hw
        )
        for i in range(self.internal_length):
            out = out + ResBlock(
                self.c_mid, self.c_mid, inner_scale, self.activation,
                c_gpool=(self.c_gpool if i == 0 else None),
                use_rvgl=self.use_rvgl,
            )(out, mask, mask_sum_hw)
        out = NormActConv(
            self.c_trunk, 1, 0.0, self.activation, use_gamma=True
        )(out, mask, mask_sum_hw)
        return out


class GoPolicyHead(nn.Module):
    """KataGo policy head (version >= 15 pass pathway), single policy output.

    Returns [N, H*W + 1] logits; the last entry is the pass move. Off-board
    positions get logit - 5000 so they vanish after softmax.
    """

    c_p1: int
    c_g1: int
    activation: str

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        act = _ACTS[self.activation]
        outp = nn.Conv(
            self.c_p1, (1, 1), use_bias=False,
            kernel_init=kata_init(0.8, self.activation),
        )(x)
        outg = nn.Conv(
            self.c_g1, (1, 1), use_bias=False,
            kernel_init=kata_init(1.0, self.activation),
        )(x)
        outg = NormMask()(outg, mask)  # BiasMask
        outg = act(outg)
        outg = kata_gpool(outg, mask, mask_sum_hw)  # [N, 3*c_g1]

        outpass = nn.Dense(
            self.c_p1, use_bias=True,
            kernel_init=kata_init(1.0, self.activation),
            bias_init=kata_init(0.2, self.activation, fan_in=3 * self.c_g1),
        )(outg)
        outpass = act(outpass)
        outpass = nn.Dense(
            1, use_bias=False, kernel_init=kata_init(0.3, "identity")
        )(outpass)  # [N, 1]

        outg = nn.Dense(
            self.c_p1, use_bias=False,
            kernel_init=kata_init(0.6, self.activation),
        )(outg)
        outp = outp + outg[:, None, None, :]
        outp = NormMask()(outp, mask)  # bias2
        outp = act(outp)
        outp = nn.Conv(
            1, (1, 1), use_bias=False, kernel_init=kata_init(0.3, "identity")
        )(outp)  # [N, H, W, 1]
        outp = outp - (1.0 - mask) * 5000.0
        n = outp.shape[0]
        return jnp.concatenate([outp.reshape(n, -1), outpass], axis=1)


class GenericPolicyHead(nn.Module):
    """A plain action-space policy head over the trunk features for non-Go games.

    This is used based on explicit game identity, not action-space shape
    (e.g. Connect4's column actions). This head has no pass-pathway/gpool
    machinery, just conv -> norm -> act -> flatten -> Dense(action_space).
    """

    action_space: int
    c_p1: int
    activation: str

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        act = _ACTS[self.activation]
        p = nn.Conv(
            self.c_p1, (1, 1), use_bias=False,
            kernel_init=kata_init(0.8, self.activation),
        )(x)
        p = NormMask()(p, mask)
        p = act(p)
        p = p.reshape(p.shape[0], -1)
        return nn.Dense(
            self.action_space,
            kernel_init=nn.initializers.normal(stddev=1e-2),
            bias_init=nn.initializers.zeros,
        )(p)


class ChessPolicyHead(nn.Module):
    """KataGo-style policy head for pgx chess's 64x73 action encoding."""

    c_p1: int
    c_g1: int
    activation: str
    num_planes: int = 73

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        act = _ACTS[self.activation]
        outp = nn.Conv(
            self.c_p1,
            (1, 1),
            use_bias=False,
            kernel_init=kata_init(0.8, self.activation),
        )(x)
        outg = nn.Conv(
            self.c_g1,
            (1, 1),
            use_bias=False,
            kernel_init=kata_init(1.0, self.activation),
        )(x)
        outg = NormMask()(outg, mask)
        outg = act(outg)
        outg = kata_gpool(outg, mask, mask_sum_hw)
        outg = nn.Dense(
            self.c_p1,
            use_bias=False,
            kernel_init=kata_init(0.6, self.activation),
        )(outg)
        outp = outp + outg[:, None, None, :]
        outp = NormMask()(outp, mask)
        outp = act(outp)
        outp = nn.Conv(
            self.num_planes,
            (1, 1),
            use_bias=False,
            kernel_init=kata_init(0.3, "identity"),
        )(outp)
        # pgx observes chess after a 90-degree rotation. Undo it so row-major
        # flattening matches action = from_square * 73 + move_plane.
        outp = jnp.rot90(outp, k=-1, axes=(1, 2))
        return outp.reshape(outp.shape[0], -1)


class ValueHead(nn.Module):
    """KataGo value head, main 3-way {win, loss, noresult} logits only."""

    c_v1: int
    c_v2: int
    activation: str

    @nn.compact
    def __call__(self, x, mask, mask_sum_hw):
        act = _ACTS[self.activation]
        v1 = nn.Conv(
            self.c_v1, (1, 1), use_bias=False,
            kernel_init=kata_init(1.0, self.activation),
        )(x)
        v1 = NormMask()(v1, mask)  # bias1
        v1 = act(v1)
        pooled = kata_value_head_gpool(v1, mask, mask_sum_hw)  # [N, 3*c_v1]
        v2 = nn.Dense(
            self.c_v2, use_bias=True,
            kernel_init=kata_init(1.0, self.activation),
            bias_init=kata_init(0.2, self.activation, fan_in=3 * self.c_v1),
        )(pooled)
        v2 = act(v2)
        return nn.Dense(
            3, use_bias=True,
            kernel_init=kata_init(1.0, "identity"),
            bias_init=kata_init(0.2, "identity", fan_in=self.c_v2),
        )(v2)


class KataGoTrunk(nn.Module):
    """The bXcYnbt trunk only (input conv + nested-bottleneck blocks + final
    norm/act), shared by the native Go head and the generic action-space head.

    ``block_gpool[i]`` says whether trunk block i is the "gpool" flavor.
    Defaults are b10c384nbt. Inputs are NHWC; off-board input features should
    be zero (they are re-masked defensively anyway).
    """

    c_trunk: int = 384
    c_mid: int = 192
    c_gpool: int = 64
    block_gpool: Sequence[bool] = (
        False, False, True, False, False, True, False, False, True, False,
    )
    internal_length: int = 2
    activation: str = "relu"
    use_rvgl: bool = True

    @nn.compact
    def __call__(
        self,
        input_spatial,  # [N, H, W, C_spatial]
        input_global=None,  # [N, C_global] or None
        mask=None,  # [N, H, W, 1] on-board mask, or None for full board
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        input_spatial = input_spatial.astype(jnp.float32)
        if mask is None:
            mask = jnp.ones_like(input_spatial[..., :1])
        mask_sum_hw = jnp.sum(mask, axis=(1, 2))  # [N, 1]

        out = nn.Conv(
            self.c_trunk, (3, 3), use_bias=False,
            kernel_init=kata_init(0.8, self.activation),
        )(input_spatial * mask)
        if input_global is not None:
            out = out + nn.Dense(
                self.c_trunk, use_bias=False,
                kernel_init=kata_init(0.6, self.activation),
            )(input_global.astype(jnp.float32))[:, None, None, :]

        fixup_scale = 1.0 / math.sqrt(len(self.block_gpool))
        for use_gpool in self.block_gpool:
            out = out + NestedBottleneckResBlock(
                self.c_trunk, self.c_mid, self.internal_length,
                fixup_scale, self.activation,
                c_gpool=(self.c_gpool if use_gpool else None),
                use_rvgl=self.use_rvgl,
            )(out, mask, mask_sum_hw)

        out = NormMask()(out, mask)  # norm_trunkfinal (fixup: bias+mask)
        out = _ACTS[self.activation](out)
        return out, mask, mask_sum_hw


def value_from_logits(value_logits: jax.Array) -> jax.Array:
    """[N, 3] {win, loss, noresult} logits -> scalar value in [-1, 1] ([N])."""
    probs = jax.nn.softmax(value_logits, axis=-1)
    return probs[..., 0] - probs[..., 1]


def _gpool_every_3(n: int) -> Tuple[bool, ...]:
    """KataGo's nbt configs put the gpool flavor on blocks 3, 6, 9, ... except
    that a final block is never gpool (b18c384nbt stops at 15)."""
    return tuple((i % 3 == 0) and (i != n) for i in range(1, n + 1))


# Trunk/head sizes lifted from modelconfigs.py (fixup base configs).
PRESETS = {
    "b5c192nbt": dict(
        c_trunk=192, c_mid=96, c_gpool=32,
        block_gpool=(False, True, False, True, False),
        c_p1=32, c_g1=32, c_v1=32, c_v2=80,
    ),
    "b10c256nbt": dict(
        c_trunk=256, c_mid=128, c_gpool=64,
        block_gpool=_gpool_every_3(10),
        c_p1=32, c_g1=32, c_v1=32, c_v2=96,
    ),
    "b10c384nbt": dict(
        c_trunk=384, c_mid=192, c_gpool=64,
        block_gpool=_gpool_every_3(10),
        c_p1=48, c_g1=48, c_v1=48, c_v2=112,
    ),
    "b18c384nbt": dict(
        c_trunk=384, c_mid=192, c_gpool=64,
        block_gpool=_gpool_every_3(18),
        c_p1=48, c_g1=48, c_v1=96, c_v2=128,
    ),
    "b28c512nbt": dict(
        c_trunk=512, c_mid=256, c_gpool=64,
        block_gpool=_gpool_every_3(28),
        c_p1=64, c_g1=64, c_v1=128, c_v2=144,
    ),
}


_DYNAMIC_PRESET_RE = re.compile(r"^b(?P<blocks>[1-9]\d*)c(?P<channels>[1-9]\d*)(?:i(?P<internal>[1-9]\d*))?nbt$")


def resolve_preset(preset: str) -> dict:
    """Resolve a fixed or dynamic KataGo-style preset string.

    Fixed presets are lifted from KataGo modelconfigs.py. Dynamic presets use
    ``b{blocks}c{channels}nbt`` and are intentionally explicit so sweeps can
    target tiny models without adding many hard-coded names. ``i{internal}``
    may be inserted before ``nbt`` to override the nested-bottleneck internal
    length; otherwise KataGo's nbt default of 2 is used.
    """
    if preset in PRESETS:
        return dict(PRESETS[preset])

    match = _DYNAMIC_PRESET_RE.match(preset)
    if not match:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"Unknown KataGo preset {preset!r}. Use a fixed preset ({valid}) "
            "or dynamic form b{blocks}c{channels}nbt, e.g. b4c16nbt."
        )

    blocks = int(match.group("blocks"))
    channels = int(match.group("channels"))
    internal = int(match.group("internal") or 2)
    head_channels = max(1, channels // 8)
    return dict(
        c_trunk=channels,
        c_mid=max(1, channels // 2),
        c_gpool=max(1, channels // 8),
        block_gpool=_gpool_every_3(blocks),
        internal_length=internal,
        c_p1=head_channels,
        c_g1=head_channels,
        c_v1=head_channels,
        c_v2=max(4, channels // 4),
    )


class KataModel(nn.Module):
    """KataGo-style network matching nanoAlphaZero's model interface:

        masked_logits, value = model(obs, valid)

    obs is a pgx NHWC observation, valid a [B, action_space] legal-move mask,
    value a scalar in [-1, 1] from the current player's perspective.
    ``deterministic`` is accepted for signature compatibility (no dropout).

    CHESS ADAPTATION: originally this always used KataGo's native Go head,
    which only ever emits H*W + 1 logits (a Go board point + one pooled pass
    move) -- fine for go, but incompatible with chess's 4672-way
    from/to/promotion move encoding. The trunk itself was never Go-specific
    (just conv + nested-bottleneck blocks over an [N,H,W,C] tensor), so it's
    now factored out into KataGoTrunk and shared by game-specific heads:
    KataGo's own GoPolicyHead for Go, a spatial 73-plane head for chess, or a
    plain flatten+Dense GenericPolicyHead otherwise. Game identity is explicit
    so an unrelated game with H*W+1 actions is not mistaken for Go.
    """

    action_space: int
    is_go: bool
    is_chess: bool = False
    c_trunk: int = 384
    c_mid: int = 192
    c_gpool: int = 64
    block_gpool: Sequence[bool] = (
        False, False, True, False, False, True, False, False, True, False,
    )
    internal_length: int = 2
    c_p1: int = 48
    c_g1: int = 48
    c_v1: int = 48
    c_v2: int = 112
    activation: str = "relu"
    use_rvgl: bool = True

    @nn.compact
    def __call__(
        self,
        obs,
        valid,
        deterministic: bool = False,
        return_wdl_logits: bool = False,
    ):
        trunk = KataGoTrunk(
            c_trunk=self.c_trunk, c_mid=self.c_mid, c_gpool=self.c_gpool,
            block_gpool=self.block_gpool, internal_length=self.internal_length,
            activation=self.activation, use_rvgl=self.use_rvgl,
        )
        out, mask, mask_sum_hw = trunk(obs)

        if self.is_go and self.is_chess:
            raise ValueError("is_go and is_chess are mutually exclusive")
        if self.is_go:
            expected_actions = out.shape[1] * out.shape[2] + 1
            if self.action_space != expected_actions:
                raise ValueError(
                    "KataGo's Go policy head requires action_space == H*W+1; "
                    f"got action_space={self.action_space}, H*W+1={expected_actions}"
                )
            # Go: board points + pass, KataGo's own two-pathway policy head.
            # Keep the historical Flax path so existing Go checkpoints load.
            policy_logits = GoPolicyHead(
                self.c_p1, self.c_g1, self.activation, name="PolicyHead_0"
            )(out, mask, mask_sum_hw)
        elif self.is_chess:
            head = ChessPolicyHead(self.c_p1, self.c_g1, self.activation)
            expected_actions = out.shape[1] * out.shape[2] * head.num_planes
            if self.action_space != expected_actions:
                raise ValueError(
                    "ChessPolicyHead requires action_space == H*W*73; "
                    f"got action_space={self.action_space}, "
                    f"H*W*73={expected_actions}"
                )
            policy_logits = head(out, mask, mask_sum_hw)
        else:
            # Other non-Go action encodings.
            policy_logits = GenericPolicyHead(
                self.action_space, self.c_p1, self.activation
            )(out, mask, mask_sum_hw)

        value_logits = ValueHead(self.c_v1, self.c_v2, self.activation)(
            out, mask, mask_sum_hw
        )
        masked_logits = jnp.where(
            valid, policy_logits, jnp.finfo(policy_logits.dtype).min
        )
        value = value_from_logits(value_logits)
        if return_wdl_logits:
            return masked_logits, value, value_logits
        return masked_logits, value

    def sample(self, logits, key, test: bool = False):
        return (
            jnp.argmax(logits, axis=-1) if test else jax.random.categorical(key, logits)
        )

def make_model(config, rng, sharding=None):
    """Build the KataGo-style network using the existing width/depth settings."""
    preset = config.get(
        "katago_preset",
        f"b{config['conv_depth']}c{config['conv_width']}nbt",
    )
    env_id = str(config["env_id"])
    model_kwargs = resolve_preset(preset)
    model = KataModel(
        action_space=config["game_num_actions"],
        is_go=env_id.startswith("go_"),
        is_chess=env_id == "chess",
        **model_kwargs,
        activation=config.get("katago_activation", "mish"),
        use_rvgl=config.get("katago_use_rvgl", True),
    )
    observation = jnp.zeros((1,) + config["game_obs_shape"])
    valid_action_mask = jnp.ones((1, config["game_num_actions"]), dtype=bool)
    model_state = init_and_shard_model(
        config, model, rng, observation, valid_action_mask, sharding
    )
    return model, model_state


# Shard model params as REPLICATED across the mesh when enabled.
def init_and_shard_model(config, model, rng, obs, valid_mask, sharding):
    use_bf16 = config.get("use_bf16", False)

    def _init_fn(rng, obs, mask):
        variables = model.init(rng, obs, mask)
        if use_bf16:
            variables = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.bfloat16), variables
            )
        return variables

    if not config.get("enable_sharding", False) or sharding is None:
        return jax.jit(_init_fn)(rng, obs, valid_mask)

    abstract_variables = jax.eval_shape(_init_fn, rng, obs, valid_mask)
    sharding_tree = jax.tree_util.tree_map(lambda _: sharding, abstract_variables)
    sharded_init = jax.jit(_init_fn, out_shardings=sharding_tree)
    with sharding.mesh:
        model_state = sharded_init(rng, obs, valid_mask)
    return model_state


# =============================================================================
# Fixed two-rung Gumbel search (unchanged production algorithm)
# Source: src/nanoalphazero/mcts.py
# =============================================================================

"""One-round sequential-halving MCTS used by nanoAlphaZero."""


# =============================================================================
# MCTS
#
# Adapted from https://github.com/google-deepmind/mctx.
#
# "1sh": a custom search that runs a single round of Sequential Halving,
# batched across all root actions. Full MCTS expands nodes one at a time; 1sh
# does the whole round in two parallel network calls:
#   1. Evaluate all `num_root_considered` root actions at once.
#   2. Keep the better half (`num_survivors`), expand one child of each, and
#      pick the best action.
# The search budget is fixed by the two rungs.
#
# Since 1sh is a fixed-shape single round, several config knobs are UNUSED --
# placeholders for swapping in MCTX's full node-by-node MCTS later:
#   - mcts_num_simulations : node-expansion budget for the full search
#   - mcts_epsilon         : qtransform epsilon
#   - mcts_max_m           : max sampled actions at the root
#   - mcts_use_gumbel      : Gumbel-MuZero vs. regular MuZero
#   - mcts_variant         : which MCTX policy to dispatch to
# =============================================================================


# Parameters are an arbitrary nested structure of chex.Array.
Params = chex.ArrayTree
Action = chex.Array
RecurrentState = Any


@chex.dataclass(frozen=True)
class RecurrentFnOutput:
    """The output of a `RecurrentFn`.

    reward: `[B]` an approximate reward from the state-action transition.
    discount: `[B]` the discount between the `reward` and the `value`.
    prior_logits: `[B, num_actions]` the logits produced by a policy network.
    value: `[B]` an approximate value of the state after the state-action
      transition.
    """

    reward: chex.Array
    discount: chex.Array
    prior_logits: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class RootFnOutput:
    """The output of a representation network.

    prior_logits: `[B, num_actions]` the logits produced by a policy network.
    value: `[B]` an approximate value of the current state.
    embedding: `[B, ...]` the inputs to the next `recurrent_fn` call.
    """

    prior_logits: chex.Array
    value: chex.Array
    embedding: RecurrentState
    k_indices: Optional[Any] = None


RecurrentFn = Callable[
    [Params, chex.PRNGKey, Action, RecurrentState],
    Tuple[RecurrentFnOutput, RecurrentState],
]


@chex.dataclass(frozen=True)
class PolicyOutput:
    """The output of a policy.

    action: `[B]` the proposed action.
    action_weights: `[B, num_actions]` the targets used to train a policy network.
    """

    action: chex.Array
    action_weights: chex.Array

    # visit counts over actions, used by the selfplay exploration sampler.
    visit_counts: Optional[Any] = None

    # BNK compressed fields (populated when use_bnk=True)
    bnk_k_indices: Optional[Any] = None
    bnk_action_weights: Optional[Any] = None


# ─────────────────────────────────────────────────────────────────────────────
# Inlined helpers from mctx._src.action_selection
# ─────────────────────────────────────────────────────────────────────────────
def _mask_invalid_actions(logits, invalid_actions):
    """Returns logits with zero mass to invalid actions."""
    if invalid_actions is None:
        return logits
    chex.assert_equal_shape([logits, invalid_actions])
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    # At the end of an episode, all actions can be invalid. A softmax would then
    # produce NaNs, if using -inf for the logits. We avoid the NaNs by using
    # a finite `min_logit` for the invalid actions.
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(invalid_actions, min_logit, logits)


def masked_argmax(to_argmax, invalid_actions):
    """Returns a valid action with the highest `to_argmax`."""
    if invalid_actions is not None:
        chex.assert_equal_shape([to_argmax, invalid_actions])
        to_argmax = jnp.where(invalid_actions, -jnp.inf, to_argmax)
    return jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Inlined qtransform helpers from mctx._src.qtransforms (doc #1)
# ─────────────────────────────────────────────────────────────────────────────
def _rescale_qvalues(qvalues, epsilon):
    """Rescales the given completed Q-values to be from the [0, 1] interval."""
    min_value = jnp.min(qvalues, axis=-1, keepdims=True)
    max_value = jnp.max(qvalues, axis=-1, keepdims=True)
    return (qvalues - min_value) / jnp.maximum(max_value - min_value, epsilon)


def _complete_qvalues(qvalues, *, visit_counts, value):
    """Returns completed Q-values, with the `value` for unvisited actions."""
    chex.assert_equal_shape([qvalues, visit_counts])
    chex.assert_shape(value, [])

    # The missing qvalues are replaced by the value.
    completed_qvalues = jnp.where(visit_counts > 0, qvalues, value)
    chex.assert_equal_shape([completed_qvalues, qvalues])
    return completed_qvalues


def _compute_mixed_value(raw_value, qvalues, visit_counts, prior_probs):
    """Interpolates the raw_value and weighted qvalues."""
    sum_visit_counts = jnp.sum(visit_counts, axis=-1)
    # Ensuring non-nan weighted_q, even if the visited actions have zero
    # prior probability.
    prior_probs = jnp.maximum(jnp.finfo(prior_probs.dtype).tiny, prior_probs)
    # Summing the probabilities of the visited actions.
    sum_probs = jnp.sum(jnp.where(visit_counts > 0, prior_probs, 0.0), axis=-1)
    weighted_q = jnp.sum(
        jnp.where(
            visit_counts > 0,
            prior_probs * qvalues / jnp.where(visit_counts > 0, sum_probs, 1.0),
            0.0,
        ),
        axis=-1,
    )
    return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)


def final_qtransform_completed_by_mix_value(
    root_qvalues,
    root_raw_value,
    root_prior_logits,
    layer1_visit_counts,
    *,
    value_scale: chex.Numeric = 1.0,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = False,
    use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
) -> chex.Array:
    """Returns the completed, transformed Q-values used to pick actions.

    The missing Q-values of the unvisited actions are replaced by the mixed
    value, defined in Appendix D of "Policy improvement by planning with
    Gumbel": https://openreview.net/forum?id=bERaNdoegnO

    The Q-values are transformed by a linear transformation:
      `(maxvisit_init + max(visit_counts)) * value_scale * qvalues`.
    """
    qvalues = root_qvalues
    visit_counts = layer1_visit_counts
    raw_value = root_raw_value
    prior_probs = jax.nn.softmax(root_prior_logits)

    # Computing the mixed value and producing completed_qvalues.
    mixed_value = _compute_mixed_value(
        raw_value, qvalues=qvalues, visit_counts=visit_counts, prior_probs=prior_probs
    )
    if use_mixed_value:
        value = mixed_value
    else:
        value = raw_value
    completed_qvalues = _complete_qvalues(
        qvalues, visit_counts=visit_counts, value=value
    )

    # Scaling the Q-values.
    rescaled_qvalues = _rescale_qvalues(completed_qvalues, epsilon)
    if rescale_values:
        completed_qvalues = rescaled_qvalues
    maxvisit = jnp.max(visit_counts, axis=-1)
    visit_scale = maxvisit_init + maxvisit

    return visit_scale * value_scale * completed_qvalues


# ─────────────────────────────────────────────────────────────────────────────
# Inlined fast-gather helpers from gumbel_muzero_policy
# ─────────────────────────────────────────────────────────────────────────────
def _fast_gather2d(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """TPU-friendly replacement for `jnp.take_along_axis(x, idx, axis=1)` on
    rank-2 tensors.

    Parameters
    ----------
    x   : [B, N]  – values to gather from
    idx : [B, K]  – int32 / int64 row indices to take (axis 1)

    Returns
    -------
    out : [B, K]  – same as the Gather version
    """
    # one-hot mask: [B, K, N]   (B=batch, K=number of indices, N=source length)
    mask = jax.nn.one_hot(idx, x.shape[1], dtype=x.dtype)  # idx : [B, K]
    out = jnp.einsum("bkn,bn->bk", mask, x)  # result [B, K]
    return out


def _fast_gather_rows(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """TPU-friendly replacement for

        jnp.take_along_axis(x, idx[..., None], axis=1)

    Works for **both** shapes

        x   : [B, N]                 (rank-2)
        x   : [B, N, F1, F2, …]      (rank ≥ 3)

    Returns out of shape [B, K, ...] – rows selected from `x`.  Trailing
    feature axes (`...`) are preserved if present.  For rank-2 input the
    result is [B, K].
    """
    if idx.ndim != 2 or idx.shape[0] != x.shape[0]:
        raise ValueError("`idx` must be [B, K] with the same batch size as `x`")
    if x.ndim < 2:
        raise ValueError("`x` must be rank ≥ 2 with the gather axis at pos 1")

    N = x.shape[1]
    # One-hot mask: [B, K, N]  (stored in x.dtype ⇒ keeps bf16/f32 throughput)
    mask = jax.nn.one_hot(idx, N, dtype=x.dtype)
    # Batched matmul:  mask[b, k, n] ⋅ x[b, n, …]  → out[b, k, …]
    out = jax.lax.dot_general(
        mask,
        x,
        (
            ((2,), (1,)),  # contract N-axis of mask with row-axis of x
            ((0,), (0,)),
        ),
    )  # keep batch axis
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main policy
# ─────────────────────────────────────────────────────────────────────────────
def gumbel_muzero_policy_1sh(
    params: Params,
    rng_key: chex.PRNGKey,
    root: RootFnOutput,
    recurrent_fn: RecurrentFn,
    *,
    num_root_considered: int = 16,  # first SH rung
    num_survivors: int = 8,  # second SH rung (num_root_considered // 2)
    gumbel_scale: chex.Numeric = 1.0,
    invalid_actions: Optional[chex.Array] = None,
    value_scale: chex.Numeric = 1.0,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = False,
    use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
    use_bnk: bool = False,
    num_k_actions: Optional[int] = None,
) -> PolicyOutput:
    """
    Sequential-Halving BFS (2 rungs).
    1.  Visit R = `num_root_considered` root actions once in parallel.
    2.  Keep the best S = `num_survivors`, visit *one* child of each of those once.
    3.  Back-up the two visits and pick the root move that maximises
        gumbel + prior + completed-Q.
    Every expansion is batched → friendly to TPU.

    Note: this policy has NO `num_simulations` knob — its search budget is fixed
    by the two rungs and controlled entirely by `num_root_considered` / `num_survivors`.
    """

    # ------------------------------------------------------------------------
    # 0) Root pre-processing
    # ------------------------------------------------------------------------
    root = root.replace(
        prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions)
    )
    B, A = root.prior_logits.shape
    R, S = num_root_considered, num_survivors  # SH rungs: R root actions, S survivors
    rng_key, g_root_key = jax.random.split(rng_key)
    root_gumbel = gumbel_scale * jax.random.gumbel(
        g_root_key, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype
    )

    # ------------------------------------------------------------------------
    # 1) FIRST RUNG  –– expand R distinct root actions once
    # ------------------------------------------------------------------------
    #    score = g + logit  (initial completed-Q is 0 so it drops out)
    first_score = root_gumbel + root.prior_logits
    _, first_idx = jax.lax.top_k(first_score, R)  # [B, R]

    # Expand root in parallel -------------------------------------------------
    BxR = B * R
    root_flat_actions = first_idx.reshape(-1)

    rng_key, _rng = jax.random.split(rng_key)
    root_flat_keys = jax.random.split(_rng, BxR).reshape(BxR, -1)
    root_flat_embed = jax.tree.map(lambda x: jnp.repeat(x, R, axis=0), root.embedding)

    layer1_flat_out, layer1_flat_emb = recurrent_fn(
        params, root_flat_keys, root_flat_actions, root_flat_embed
    )

    def unflat(x):
        return x.reshape(B, R, *x.shape[1:])

    layer1_out = jax.tree.map(unflat, layer1_flat_out)  # [BxR,] -> [B, R]
    layer1_embeds = jax.tree.map(unflat, layer1_flat_emb)
    layer1_qvalues = layer1_out.reward + layer1_out.discount * layer1_out.value
    layer1_visits = jnp.ones_like(layer1_qvalues, dtype=jnp.int32)  # visits = 1

    # ------------------------------------------------------------------
    # 1-bis)  Mask out actions that are invalid at the root
    # ------------------------------------------------------------------
    if invalid_actions is not None:
        # valid_mask : 1 for legal actions, 0 for invalid
        layer1_valid_mask = 1 - _fast_gather2d(invalid_actions, first_idx)  # [B,R]

        layer1_qvalues = layer1_qvalues * layer1_valid_mask
        layer1_visits = layer1_visits * layer1_valid_mask.astype(
            layer1_qvalues.dtype
        )  # visits = 0 for invalid

        # actions that are invalid should never survive to rung-2
        # set their score to −inf so top_k ignores them
        layer1_score_mask = layer1_valid_mask == 0
    else:
        layer1_score_mask = jnp.zeros_like(layer1_qvalues, dtype=bool)

    # 1-fin) Calc the completed_qvalues
    def layer1_qtransform(q1):
        alpha = value_scale * (maxvisit_init + 1.0)  # same scale as paper
        if rescale_values:
            q_min = jnp.min(q1, axis=1, keepdims=True)
            q_max = jnp.max(q1, axis=1, keepdims=True)
            q_norm = (q1 - q_min) / jnp.maximum(q_max - q_min, epsilon)
        else:
            q_norm = q1  # no rescaling
        cq = alpha * q_norm  # completed-Q for the R parents
        return cq, q_norm

    # If we visit and the value is negative, we should pick that over invalid action
    # This will happen when we do top_k with masked_score
    layer1_cqvalues, _ = layer1_qtransform(layer1_qvalues)

    # ------------------------------------------------------------------------
    # 2) SECOND RUNG  –– keep best S roots, add one extra rollout inside each
    # ------------------------------------------------------------------------
    # score_after_1 = g + logit + q1
    score1 = (
        jnp.take_along_axis(root_gumbel, first_idx, -1)
        + jnp.take_along_axis(root.prior_logits, first_idx, -1)
        + layer1_cqvalues
    )

    masked_score1 = jnp.where(layer1_score_mask, -jnp.inf, score1)
    _, second_loc = jax.lax.top_k(masked_score1, S)  # [B, S] the idx within the R
    second_idx = _fast_gather2d(first_idx, second_loc)  # [B, S]

    # 2-b) which of those S parents were illegal to begin with? --------------
    illegal_parent = _fast_gather2d(layer1_score_mask, second_loc)  # [B,S] Bool

    # Expand *one child* of each of those S parents in layer 1 ---------------
    # Gather the chosen parents' logits so we can pick a child
    layer1_halved_logits = jnp.take_along_axis(
        layer1_out.prior_logits,  # [B,R,A]
        second_loc[..., None],
        1,
    )

    # completed-Q values for each of the S parents (all children unvisited → 0)
    layer1_halved_completed_q = jnp.zeros_like(layer1_halved_logits)

    # Apply the interior selection heuristic once, batched over [B,S]
    probs = jax.nn.softmax(layer1_halved_logits + layer1_halved_completed_q, axis=-1)
    to_argmax = probs  # since visits=0
    best_child = jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)  # [B,S]

    # Flatten and expand those leaf actions
    BxS = B * S
    leaf_actions = best_child.reshape(-1)

    rng_key, key_leaf = jax.random.split(rng_key)
    leaf_keys = jax.random.split(key_leaf, BxS).reshape(BxS, -1)

    def gather_parents_leaf(x: jnp.ndarray) -> jnp.ndarray:
        """
        Pick the S survivors (rows indexed by `second_loc`) from the R parents
        and flatten to [B*S, …].  Works for rank-2 and rank-≥3 tensors.
        """
        picked = _fast_gather_rows(x, second_loc)  # [B, S, …] or [B, S]
        return picked.reshape(BxS, *x.shape[2:])  # flatten first two axes

    layer2_parent_emb_flat = jax.tree.map(gather_parents_leaf, layer1_embeds)

    flat2_out, _ = recurrent_fn(
        params, leaf_keys, leaf_actions, layer2_parent_emb_flat
    )  # [BxS]

    def unflat2(x):
        return x.reshape(B, S, *x.shape[1:])

    layer2 = jax.tree.map(unflat2, flat2_out)  # [B, S]

    # 2-e) compute q₂ only for *legal* parents --------------------------------
    q2_leaf = layer2.reward + layer2.discount * layer2.value  # [B,S]

    # ------------------------------------------------------------------
    # (a) rewards and discounts of the S survivors  (rank-2)
    # ------------------------------------------------------------------
    r1 = _fast_gather2d(layer1_out.reward, second_loc)  # [B,S]
    γ1 = _fast_gather2d(layer1_out.discount, second_loc)  # [B,S]
    q2_full = r1 + γ1 * q2_leaf  # [B,S]

    # ------------------------------------------------------------------
    # (b) q / visit counts from the first rung that correspond to the
    #     same S survivors                                      (rank-2)
    # ------------------------------------------------------------------
    q1_sel = _fast_gather2d(layer1_qvalues, second_loc)  # [B,S]
    v1_sel = _fast_gather2d(layer1_visits, second_loc)  # [B,S]

    # mask-out the illegal parents: keep their original q₁, no extra visit
    q2 = jnp.where(illegal_parent, 0.0, q2_full)  # [B,S]
    v2 = jnp.where(illegal_parent, 0, 1).astype(jnp.int32)  # [B,S]

    q_comb = (q1_sel * v1_sel + q2) / (v1_sel + v2 + 1e-6)  # [B,S]
    vcnt2 = v1_sel + v2  # [B,S]

    # ------------------------------------------------------------------------
    # 3) [optimized] Assemble per-action arrays for the root (q & visit-count)
    # ------------------------------------------------------------------------
    #  – layer-1 contribution ………………   first_idx,     layer1_qvalues / layer1_visits
    #  – layer-2 overwrite   ………………   second_idx,    q_comb        / vcnt2
    #    (second_idx ⊂ first_idx, so we "mask-away & add" to overwrite)

    # One-hot masks -----------------------------------------------------------
    mask1 = jax.nn.one_hot(first_idx, A)  # [B, K₁, A]
    mask2 = jax.nn.one_hot(second_idx, A)  # [B, K₂, A]

    # Σ mask * value  →  [B, A] -----------------------------
    q_l1 = jnp.sum(mask1 * layer1_qvalues[:, :, None], 1)  # first-rung q
    v_l1 = jnp.sum(mask1 * layer1_visits[:, :, None], 1)

    q_l2 = jnp.sum(mask2 * q_comb[:, :, None], 1)  # second-rung q
    v_l2 = jnp.sum(mask2 * vcnt2[:, :, None], 1)

    mask2_sum = jnp.sum(mask2, axis=1)  # [B, A]  1 on survivors

    # Overwrite: zero-out the survivors in layer-1 arrays,
    # then add layer-2 values -----------------------------------------------
    q_root = q_l1 * (1 - mask2_sum) + q_l2  # [B, A]
    visit_root = v_l1 * (1 - mask2_sum) + v_l2.astype(v_l1.dtype)

    # ------------------------------------------------------------------------
    # 4) Completed-Q transform & final root decision
    # ------------------------------------------------------------------------
    qtransform_fn = functools.partial(
        final_qtransform_completed_by_mix_value,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
        rescale_values=rescale_values,
        use_mixed_value=use_mixed_value,
        epsilon=epsilon,
    )

    completed_q = jax.vmap(qtransform_fn, in_axes=[0, 0, 0, 0])(
        q_root, root.value, root.prior_logits, visit_root
    )

    final_score = root_gumbel + root.prior_logits + completed_q
    best_a = masked_argmax(final_score, invalid_actions)

    search_logits = root.prior_logits + completed_q

    # Final mask to ensure invalid actions are -inf
    search_logits = _mask_invalid_actions(search_logits, invalid_actions)
    action_weights = jax.nn.softmax(search_logits)

    # BNK compressed fields: top-k over the full search_logits so we capture
    # completed-Q info for all A actions (not just the num_root_considered explored ones).
    if use_bnk:
        _, bnk_k_indices = jax.lax.top_k(search_logits, k=num_k_actions)  # [B, K]
        k_search_logits = _fast_gather2d(search_logits, bnk_k_indices)  # [B, K]
        bnk_action_weights = jax.nn.softmax(k_search_logits)  # [B, K]
    else:
        bnk_k_indices = None
        bnk_action_weights = None

    return PolicyOutput(
        # --- decision & training targets ---
        action=best_a,  # int32  [B]
        action_weights=action_weights,  # float [B, A]
        visit_counts=visit_root,  # [B, A]
        # BNK compressed fields (populated when use_bnk=True)
        bnk_k_indices=bnk_k_indices,  # [B, K1] or None
        bnk_action_weights=bnk_action_weights,  # [B, K1] or None
    )


def make_mcts(config, wenv, model, data_sharding=None):
    is_chess = config["env_id"] == "chess"
    if config.get("enable_sharding", False) and data_sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), "x")
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x")
        )

    # custom pgx chess exposes legal as packed uint32 bitmask
    #     (legal_action_bitmask), not legal_action_mask. Unpack on read.
    def _legal_from_state(env_state):
        if is_chess:
            return unpack_bitmask_vmap(env_state.legal_action_bitmask)
        return env_state.legal_action_mask

    def get_root_fn(params):
        def root_fn(env_state, _rng_key: chex.PRNGKey) -> RootFnOutput:
            obs = wenv.observe(env_state, env_state.current_player)
            legal = _legal_from_state(env_state)
            # pin obs/legal to data-parallel sharding
            if config.get("enable_sharding", False):
                obs = jax.lax.with_sharding_constraint(obs, data_sharding)
                legal = jax.lax.with_sharding_constraint(legal, data_sharding)
            model_state = {"params": params}
            prior_logits, value = model.apply(model_state, obs, legal)
            # KataGo root-temperature softening (good for chess)
            if config.get("exp_use_root_temperature", False):
                tau = config.get("exp_root_temperature", 1.3)
                prior_logits = jnp.where(
                    legal, prior_logits / tau, jnp.finfo(prior_logits.dtype).min
                )
            return RootFnOutput(
                prior_logits=prior_logits,
                value=value,
                embedding=env_state,
            )

        return root_fn

    def get_recurrent_fn():
        def recurrent_fn(params, rng_key, action, env_state):
            action = jnp.asarray(action, dtype=jnp.int32)
            prev_player = env_state.current_player

            if config.get("enable_sharding", False):
                action = jax.lax.with_sharding_constraint(
                    action, data_sharding
                )
                prev_player = jax.lax.with_sharding_constraint(
                    prev_player, data_sharding
                )

            # 1sh uses a single rng_key (not batched)
            env_state = wenv.autostep(env_state, action, rng_key)

            obs = wenv.observe(env_state, env_state.current_player)
            legal = _legal_from_state(env_state)
            model_state = {"params": params}
            prior_logits, value = model.apply(model_state, obs, legal)

            B = env_state.rewards.shape[0]
            reward = env_state.rewards[jnp.arange(B), prev_player]
            discount = jnp.where(env_state.terminated, 0, -1).astype(jnp.float32)
            final_value = jnp.where(env_state.terminated, 0, value).astype(jnp.float32)

            recurrent_fn_output = RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=prior_logits,
                value=final_value,
            )
            return recurrent_fn_output, env_state

        return recurrent_fn

    @functools.partial(jax.jit, static_argnums=(4, 5))
    def run_mcts(
        rng_key: chex.PRNGKey,
        env_state,
        params,
        gumbel_scale,
        batch_size,
        num_simulations=config["mcts_num_simulations"],  # unused by 1sh; see note above
    ):
        key1, key2 = jax.random.split(rng_key)
        root_fn = get_root_fn(params)
        root = root_fn(env_state, jax.random.split(key2, batch_size))

        recurrent_fn = get_recurrent_fn()
        # chess uses bitmask + unpack; others use bool mask
        if is_chess:
            invalid_actions = ~unpack_bitmask_vmap(env_state.legal_action_bitmask)
        else:
            invalid_actions = ~env_state.legal_action_mask

        policy_output = gumbel_muzero_policy_1sh(
            params=params,
            invalid_actions=invalid_actions,
            rng_key=key2,
            root=root,
            recurrent_fn=recurrent_fn,
            gumbel_scale=gumbel_scale,
            value_scale=config["mcts_value_scale"],
            rescale_values=config["mcts_rescale_values"],
            maxvisit_init=config["mcts_maxvisit_init"],
            num_root_considered=config["mcts_num_root_considered"],
            num_survivors=config["mcts_num_survivors"],  # just num_root_considered // 2
            use_bnk=config.get("exp_bnk_action_weights", False),
            num_k_actions=config.get("mcts_num_k_actions", None),
        )

        return policy_output

    return run_mcts


# =============================================================================


# =============================================================================
# AlphaZero reference checkpoint format and path encoding
# Source: src/nanoalphazero/checkpoint.py
# =============================================================================

"""Safetensors checkpoint paths, metadata, saving, and loading."""


# =============================================================================
# Checkpointing
# =============================================================================
def default_ckpt_path(env_name: str) -> str:
    """Default on-disk location for a saved alphazero checkpoint."""
    return os.path.join("artifacts", f"alphazero_{env_name}.safetensors")


_CHECKPOINT_FORMAT = "nanoalphazero.flax.params"
_CHECKPOINT_FORMAT_VERSION = "2"
_TREE_PATH_ENCODING = "json-pointer-segments-v1"
_CHECKPOINT_CONFIG_KEYS = (
    "katago_preset",
    "katago_activation",
    "katago_use_rvgl",
    "use_wdl",
    "env_id",
    "game_obs_shape",
    "game_num_actions",
)


def checkpoint_model_config(config: dict) -> dict:
    """Return the resolved architecture/game settings needed to load params."""
    resolved = {
        key: config[key] for key in _CHECKPOINT_CONFIG_KEYS if key in config
    }
    resolved.update(
        {
            "katago_preset": config.get(
                "katago_preset",
                f"b{config['conv_depth']}c{config['conv_width']}nbt",
            ),
            "katago_activation": config.get("katago_activation", "mish"),
            "katago_use_rvgl": config.get("katago_use_rvgl", True),
            "use_wdl": config.get("use_wdl", True),
        }
    )
    return resolved


def apply_checkpoint_model_config(config: dict, model_config: Optional[dict]) -> dict:
    """Overlay only model/game compatibility settings from a checkpoint."""
    if not model_config:
        return config
    updated = config.copy()
    updated.update(
        {
            key: value
            for key, value in model_config.items()
            if key in _CHECKPOINT_CONFIG_KEYS
        }
    )
    if updated.get("game_obs_shape") is not None:
        updated["game_obs_shape"] = tuple(updated["game_obs_shape"])
    return updated


def _encode_path_segment(segment: Any) -> str:
    return str(segment).replace("~", "~0").replace("/", "~1")


def _decode_path_segment(segment: str) -> str:
    return segment.replace("~1", "/").replace("~0", "~")


def _flatten_checkpoint_params(params) -> dict[str, np.ndarray]:
    flat = flax.traverse_util.flatten_dict(params)
    encoded = {}
    for path, value in flat.items():
        name = "/".join(_encode_path_segment(segment) for segment in path)
        if not name or name in encoded:
            raise ValueError(f"Duplicate or empty encoded parameter path: {path!r}")
        encoded[name] = np.ascontiguousarray(jax.device_get(value))
    return dict(sorted(encoded.items()))


def _unflatten_checkpoint_params(flat_params: dict[str, jax.Array]) -> dict:
    decoded = {}
    for name, value in flat_params.items():
        path = tuple(_decode_path_segment(segment) for segment in name.split("/"))
        if not name or path in decoded:
            raise ValueError(f"Duplicate or empty parameter path in checkpoint: {name!r}")
        decoded[path] = value
    return flax.traverse_util.unflatten_dict(decoded)


def _require_safetensors_path(path: str) -> None:
    if not path.endswith(".safetensors"):
        raise ValueError("Checkpoint path must end in .safetensors")


def save_checkpoint(params, config: dict, path: str) -> None:
    """Atomically save params and resolved model metadata as Safetensors."""
    _require_safetensors_path(path)
    directory = os.path.dirname(path)
    target_dir = directory or "."
    os.makedirs(target_dir, exist_ok=True)
    metadata = {
        "format": _CHECKPOINT_FORMAT,
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "tree_path_encoding": _TREE_PATH_ENCODING,
        "model_config": json.dumps(
            checkpoint_model_config(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    }
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=target_dir
    )
    os.close(fd)
    try:
        save_safetensors_file(
            _flatten_checkpoint_params(params), temporary_path, metadata=metadata
        )
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f"✅ Saved model params to {path}")


def load_checkpoint(path: str):
    """Load Safetensors params and return `(params, model_config)`."""
    _require_safetensors_path(path)
    if not os.path.exists(path):
        raise SystemExit(
            f"No checkpoint found at {path}. Train a model first, or point "
            f"--load at an existing checkpoint."
        )
    with safe_open(path, framework="flax") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported checkpoint format in {path}")
        if metadata.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version "
                f"{metadata.get('format_version')!r} in {path}"
            )
        if metadata.get("tree_path_encoding") != _TREE_PATH_ENCODING:
            raise ValueError(f"Unsupported parameter path encoding in {path}")
        try:
            model_config = json.loads(metadata["model_config"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid or missing model_config in {path}") from exc
        flat_params = {
            name: checkpoint.get_tensor(name) for name in checkpoint.keys()
        }
    params = _unflatten_checkpoint_params(flat_params)
    print(f"✅ Loaded model params from {path}")
    return params, model_config


# =============================================================================
# Real environment adapter: actual transitions only
# Source: src/nanoalphazero/core.py
# =============================================================================


@dataclass
class WrappedEnv:
    obs_shape: Tuple
    num_actions: int
    init: Callable
    step: Callable
    autostep: Callable
    init_dummy_estate: Callable
    single_estate: Any
    observe: Optional[Callable] = None
    replay_batch: Optional[Callable] = None

    def __repr__(self) -> str:
        return f"WrappedEnv(obs_shape={self.obs_shape}, num_actions={self.num_actions})"


def make_env(config):
    env_id = config["env_id"]
    # chess carries legality as a packed uint32 bitmask; see make_mcts
    env_kwargs = {"use_bitmask": True} if env_id == "chess" else {}
    env = pgx1.make(env_id, **env_kwargs)
    e_step = env.step
    a_step = auto_reset(e_step, env.init)
    vmap_env_init = jax.jit(jax.vmap(env.init))
    vmap_env_step = jax.jit(jax.vmap(e_step))
    vmap_auto_step = jax.jit(jax.vmap(a_step))

    single_estate = env.init(jax.random.PRNGKey(0))

    def init_dummy_estate(batch_size: int):
        rng_key = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng_key, batch_size)
        return vmap_env_init(rng_keys)

    batch_size = 1
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    env_state = vmap_env_init(keys)

    vmap_observe_fn = jax.jit(jax.vmap(env.observe))
    es_obs = vmap_observe_fn(env_state, env_state.current_player)

    pgx_num_actions = env.num_actions
    pgx_obs_shape = jnp.squeeze(es_obs, axis=0).shape

    return WrappedEnv(
        obs_shape=pgx_obs_shape,
        num_actions=pgx_num_actions,
        init=vmap_env_init,
        step=vmap_env_step,
        autostep=vmap_auto_step,
        init_dummy_estate=init_dummy_estate,
        single_estate=single_estate,
        observe=vmap_observe_fn,
        replay_batch=getattr(env, "replay_batch", None),
    )


# =============================================================================
# Opening reference tables: diagnostics only
# Source: src/nanoalphazero/eval/hex/perfect_play.py
# =============================================================================

"""Solved first-move outcomes for the supported Hex boards."""


P1_WINNING_OPENINGS: dict[int, tuple[int, ...]] = {
    4: (3, 6, 9, 12),
    5: (4, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20),
    6: (
        5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 30,
    ),
    7: (
        6, 9, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25,
        26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 39, 42,
    ),
    8: (
        7, 14, 15, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48,
        49, 56,
    ),
    9: (
        8, 9, 10, 11, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 28,
        29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59,
        60, 61, 63, 64, 69, 70, 71, 72,
    ),
}


def p1_winning_openings(board_size: int) -> tuple[int, ...] | None:
    return P1_WINNING_OPENINGS.get(int(board_size))


def perfect_play_values(board_size: int) -> np.ndarray | None:
    winning = p1_winning_openings(board_size)
    if winning is None:
        return None
    values = np.ones(board_size * board_size, dtype=np.float32)
    values[np.asarray(winning, dtype=np.int32)] = -1.0
    return values.reshape((board_size, board_size))


def opening_metrics(board_size: int, records: list[dict]) -> dict[str, int | float]:
    required = set(p1_winning_openings(board_size) or ())
    model_wins = {
        int(record["opening_action"])
        for record in records
        if record.get("winner") == 0
    }
    won = len(required & model_wins)
    total = len(required)
    return {
        "perfect_opening_wins": won,
        "perfect_opening_total": total,
        "perfect_opening_fraction": won / total if total else float("nan"),
        "perfect_play_achieved": int(bool(total) and won == total),
        "unexpected_opening_wins": len(model_wins - required),
    }


# =============================================================================
# Optional external MoHex engine management
# Source: src/nanoalphazero/eval/hex/engine.py
# =============================================================================

"""Pure-stdlib MoHex GTP process management.

This module intentionally does not import JAX.  Callers can create the engine
bank before libtpu is initialized.
"""


# The default engine configuration is embedded in this script.


def action_to_vertex(action: int, board_size: int) -> str:
    if action < 0 or action >= board_size * board_size:
        raise ValueError(f"Hex action out of range: {action}")
    row, column = divmod(int(action), int(board_size))
    return f"{chr(ord('a') + column)}{row + 1}"


def vertex_to_action(vertex: str, board_size: int) -> int:
    value = vertex.strip().lower()
    if len(value) < 2 or not value[0].isalpha():
        raise ValueError(f"invalid Hex vertex from engine: {vertex!r}")
    column = ord(value[0]) - ord("a")
    try:
        row = int(value[1:]) - 1
    except ValueError as exc:
        raise ValueError(f"invalid Hex vertex from engine: {vertex!r}") from exc
    if not (0 <= row < board_size and 0 <= column < board_size):
        raise ValueError(f"Hex vertex out of range: {vertex!r}")
    return row * board_size + column


def resolve_executable(value: str) -> Path:
    expanded = Path(value).expanduser()
    if expanded.is_absolute() or expanded.parent != Path("."):
        path = expanded.resolve()
    else:
        found = shutil.which(value)
        if found is None:
            raise FileNotFoundError(f"MoHex executable not found: {value}")
        path = Path(found).resolve()
    if not path.is_file() or not path.stat().st_mode & 0o111:
        raise FileNotFoundError(f"MoHex executable not found or not executable: {path}")
    return path


def resolve_config(value: str | None) -> Path:
    if value in (None, "", "default"):
        return standalone_mohex_config()
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MoHex config not found: {path}")
    return path


class HexEngine(Protocol):
    def clear(self) -> None: ...
    def play(self, color: str, action: int) -> None: ...
    def begin_genmove(self, color: str) -> None: ...
    def finish_genmove(self) -> int | None: ...
    def close(self) -> None: ...


class MoHexEngine:
    def __init__(self, executable: Path, config: Path, board_size: int, seed: int):
        self.board_size = int(board_size)
        # Never pipe stderr: solver diagnostics can otherwise fill the pipe while
        # the synchronous GTP reader waits on stdout. Keep it for crash reports.
        self.stderr = tempfile.TemporaryFile(mode="w+t")
        self.process = subprocess.Popen(
            [
                str(executable),
                "--quiet",
                "--use-logfile=0",
                f"--config={config}",
                f"--seed={int(seed)}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        try:
            self.query(f"boardsize {self.board_size}")
        except BaseException:
            self.close()
            raise

    def _write(self, command: str) -> None:
        if self.process.poll() is not None or self.process.stdin is None:
            raise IOError(f"MoHex exited before command {command!r}")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _answer(self) -> str:
        if self.process.stdout is None:
            raise IOError("MoHex stdout is unavailable")
        lines: list[str] = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                self.stderr.seek(0)
                stderr = self.stderr.read()
                raise IOError(
                    f"MoHex exited unexpectedly with code {self.process.poll()}: {stderr.strip()}"
                )
            if line in ("\n", "\r\n"):
                break
            lines.append(line.rstrip("\r\n"))
        if not lines:
            raise IOError("MoHex returned an empty GTP response")
        marker = lines[0][:1]
        payload = "\n".join([lines[0][1:].lstrip(), *lines[1:]]).strip()
        if marker != "=":
            raise IOError(f"MoHex GTP error: {payload or lines[0]}")
        return payload

    def query(self, command: str) -> str:
        self._write(command)
        return self._answer()

    def clear(self) -> None:
        self.query("clear_board")

    def play(self, color: str, action: int) -> None:
        self.query(f"play {color} {action_to_vertex(action, self.board_size)}")

    def begin_genmove(self, color: str) -> None:
        # Benzene's reg_genmove does not update the board.  The match runner
        # validates and then sends the returned move through play().
        self._write(f"reg_genmove {color}")

    def finish_genmove(self) -> int | None:
        response = self._answer().strip().lower()
        if response == "resign":
            return None
        if response == "pass":
            raise ValueError("MoHex returned pass in a no-pass Hex game")
        return vertex_to_action(response, self.board_size)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._write("quit")
                self.process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self.stderr.close()


class MoHexPool:
    """One engine per real opening; padded JAX rows never reach this pool."""

    kind = "mohex"

    def __init__(
        self,
        path: str,
        config: str | None,
        board_size: int,
        count: int,
        *,
        seed: int = 1,
        engine_factory=MoHexEngine,
    ):
        self.executable = resolve_executable(path)
        self.config = resolve_config(config)
        self.board_size = int(board_size)
        self.engines: list[HexEngine] = []
        try:
            for index in range(int(count)):
                self.engines.append(
                    engine_factory(
                        self.executable, self.config, self.board_size, seed + index
                    )
                )
        except BaseException:
            self.close()
            raise

    def reset(self, openings: Sequence[int]) -> None:
        if len(openings) != len(self.engines):
            raise ValueError("opening count does not match the MoHex engine bank")
        for engine, opening in zip(self.engines, openings, strict=True):
            engine.clear()
            engine.play("b", int(opening))

    def play(self, color: str, actions: Sequence[int], active: Sequence[bool]) -> None:
        for engine, action, is_active in zip(
            self.engines, actions, active, strict=True
        ):
            if is_active:
                engine.play(color, int(action))

    def generate(self, color: str, active: Sequence[bool]) -> list[int | None]:
        selected = [
            (index, engine)
            for index, (engine, is_active) in enumerate(
                zip(self.engines, active, strict=True)
            )
            if is_active
        ]
        for _, engine in selected:
            engine.begin_genmove(color)
        output: list[int | None] = [-1] * len(self.engines)
        with ThreadPoolExecutor(max_workers=max(1, len(selected))) as executor:
            futures = {
                executor.submit(engine.finish_genmove): index
                for index, engine in selected
            }
            for future, index in futures.items():
                output[index] = future.result()
        return output

    def close(self) -> None:
        for engine in self.engines:
            engine.close()
        self.engines.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# =============================================================================
# Real-game MoHex evaluation
# Source: src/nanoalphazero/eval/hex/runtime.py
# =============================================================================

"""Batched Hex model-vs-MoHex match runtime."""


def _physical_size(real_size: int, *, sharded: bool) -> int:
    if not sharded:
        return real_size
    devices = max(1, jax.device_count())
    return real_size + (-real_size % devices)


def _record_move(history: list[dict[str, Any]], color: str, action: int, size: int):
    history.append(
        {
            "color": color,
            "action": int(action),
            "vertex": action_to_vertex(int(action), size),
        }
    )


def run_match(run_mcts_fn, wenv, config, params, engine_pool, *, seed: int = 1):
    """Play one real game for every possible P1 opening."""
    size = int(config["boardsize"])
    real_size = size * size
    if len(engine_pool.engines) != real_size:
        raise ValueError(
            f"Hex {size}x{size} requires {real_size} MoHex engines; "
            f"got {len(engine_pool.engines)}"
        )
    sharded = bool(config.get("enable_sharding", False))
    physical_size = _physical_size(real_size, sharded=sharded)
    openings = np.arange(real_size, dtype=np.int32)
    physical_openings = np.pad(openings, (0, physical_size - real_size))

    state = wenv.init_dummy_estate(physical_size)
    state = wenv.step(state, jnp.asarray(physical_openings))
    if sharded:

        state = jax.device_put(state, standalone_data_sharding())

    engine_pool.reset(openings.tolist())
    histories: list[list[dict[str, Any]]] = [[] for _ in range(real_size)]
    for opening, history in zip(openings, histories, strict=True):
        _record_move(history, "b", int(opening), size)

    active = np.ones(real_size, dtype=np.bool_)
    records: list[dict[str, Any] | None] = [None] * real_size
    model_seconds = 0.0
    engine_seconds = 0.0
    started = time.perf_counter()
    current_player = 1

    # One opening stone is already on the board. Hex must finish within N^2 moves.
    for ply in range(1, real_size):
        if not active.any():
            break
        legal = np.asarray(jax.device_get(state.legal_action_mask))[:real_size]
        physical_actions = np.zeros(physical_size, dtype=np.int32)

        if current_player == 0:
            tic = time.perf_counter()
            output = run_mcts_fn(
                jax.random.PRNGKey(np.uint32(seed + ply)),
                state,
                params,
                0.0,
                physical_size,
            )
            physical_actions[:] = np.asarray(
                jax.device_get(output.action), dtype=np.int32
            )
            model_seconds += time.perf_counter() - tic
            real_actions = physical_actions[:real_size]
            for index in np.flatnonzero(active):
                action = int(real_actions[index])
                if not legal[index, action]:
                    raise ValueError(
                        f"model returned illegal action {action} in opening {index}"
                    )
                _record_move(histories[index], "b", action, size)
            engine_pool.play("b", real_actions.tolist(), active.tolist())
        else:
            tic = time.perf_counter()
            generated = engine_pool.generate("w", active.tolist())
            engine_seconds += time.perf_counter() - tic
            play_active = active.copy()
            real_actions = np.zeros(real_size, dtype=np.int32)
            for index in np.flatnonzero(active):
                action = generated[index]
                if action is None:
                    records[index] = _game_record(
                        index,
                        histories[index],
                        winner=0,
                        termination="mohex_resignation",
                    )
                    active[index] = False
                    play_active[index] = False
                    continue
                action = int(action)
                if not legal[index, action]:
                    raise ValueError(
                        f"MoHex returned illegal action {action} in opening {index}"
                    )
                real_actions[index] = action
                physical_actions[index] = action
                _record_move(histories[index], "w", action, size)
            engine_pool.play("w", real_actions.tolist(), play_active.tolist())

        state = wenv.step(state, jnp.asarray(physical_actions))
        terminated = np.asarray(jax.device_get(state.terminated))[:real_size]
        rewards = np.asarray(jax.device_get(state.rewards))[:real_size]
        for index in np.flatnonzero(active & terminated):
            winner = 0 if rewards[index, 0] > rewards[index, 1] else 1
            records[index] = _game_record(
                index, histories[index], winner=winner, termination="normal"
            )
            active[index] = False
        current_player = 1 - current_player

    for index in np.flatnonzero(active):
        records[index] = _game_record(
            index, histories[index], winner=None, termination="unscored_ply_cap"
        )
    completed = [record for record in records if record is not None]
    if len(completed) != real_size:
        raise RuntimeError("Hex evaluation did not produce one record per opening")
    elapsed = time.perf_counter() - started
    wins = sum(record["winner"] == 0 for record in completed)
    losses = sum(record["winner"] == 1 for record in completed)
    unscored = real_size - wins - losses
    summary = {
        "board_size": size,
        "real_games": real_size,
        "physical_batch_size": physical_size,
        "padding_rows": physical_size - real_size,
        "model_wins": wins,
        "model_losses": losses,
        "unscored": unscored,
        "model_win_rate": wins / real_size,
        "total_plies": sum(int(record["plies"]) for record in completed),
        "elapsed_seconds": elapsed,
        "model_search_seconds": model_seconds,
        "mohex_search_seconds": engine_seconds,
        **opening_metrics(size, completed),
    }
    return completed, summary


def _game_record(opening: int, moves: list[dict], *, winner, termination: str):
    return {
        "opening_action": int(opening),
        "opening_vertex": moves[0]["vertex"],
        "winner": winner,
        "model_won": winner == 0,
        "termination": termination,
        "plies": len(moves),
        "moves": list(moves),
    }


def training_metrics(summary: dict[str, Any]) -> dict[str, int | float]:
    keys = (
        "real_games",
        "model_wins",
        "model_losses",
        "unscored",
        "model_win_rate",
        "elapsed_seconds",
        "model_search_seconds",
        "mohex_search_seconds",
        "perfect_opening_wins",
        "perfect_opening_total",
        "perfect_opening_fraction",
        "perfect_play_achieved",
        "unexpected_opening_wins",
    )
    return {f"hex_eval/{key}": summary[key] for key in keys}


def print_grid(records: list[dict], board_size: int) -> None:
    cells = {
        int(record["opening_action"]): (
            "W" if record["winner"] == 0 else "L" if record["winner"] == 1 else "?"
        )
        for record in records
    }
    print(f"\n--- Hex {board_size}x{board_size} model (P1) vs MoHex ---")
    for row in range(board_size):
        print("  " + " ".join(cells[row * board_size + col] for col in range(board_size)))


# =============================================================================
# Periodic MoHex evaluation and raw result logging
# Source: src/nanoalphazero/eval/hex/training.py
# =============================================================================

"""Training adapter and local observability for Hex engine evaluations."""


SCHEMA_VERSION = 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _opening_results(records) -> list[dict]:
    return [
        {
            "opening_action": int(record["opening_action"]),
            "opening_vertex": record["opening_vertex"],
            "result": (
                "W" if record["winner"] == 0 else "L" if record["winner"] == 1 else "?"
            ),
        }
        for record in records
    ]


class HexTrainingEvaluator:
    def __init__(self, period: int, engine_pool, output_dir, board_size: int):
        self.period = int(period)
        self.engine_pool = engine_pool
        self.output_dir = Path(output_dir)
        self.board_size = int(board_size)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(
            self.output_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "state": "waiting",
                "board_size": self.board_size,
                "games_total": self.board_size * self.board_size,
                "period": self.period,
                "timestamp": _timestamp(),
            },
        )
        print(f"Hex evaluation data: {self.output_dir}", flush=True)

    def run_if_due(self, cycle, run_mcts, env, config, params, *, train_step=None):
        if not self.period or cycle % self.period:
            return {}

        cycle = int(cycle)
        common = {
            "schema_version": SCHEMA_VERSION,
            "cycle": cycle,
            "train_step": None if train_step is None else int(train_step),
            "board_size": self.board_size,
            "games_total": self.board_size * self.board_size,
        }
        _atomic_write_json(
            self.output_dir / "status.json",
            {**common, "state": "running", "timestamp": _timestamp()},
        )
        try:
            records, summary = run_match(
                run_mcts,
                env,
                config,
                params,
                self.engine_pool,
                seed=1234 + cycle,
            )
        except BaseException as error:
            _atomic_write_json(
                self.output_dir / "status.json",
                {
                    **common,
                    "state": "failed",
                    "timestamp": _timestamp(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            print(
                f"HEX_EVAL_FAILED cycle={cycle} error={type(error).__name__}",
                flush=True,
            )
            raise

        result = {
            **common,
            **summary,
            "state": "complete",
            "timestamp": _timestamp(),
            "opening_results": _opening_results(records),
        }
        _atomic_write_jsonl(
            self.output_dir / f"games-cycle-{cycle:06d}.jsonl", records
        )
        _atomic_write_json(self.output_dir / "latest.json", result)
        _append_jsonl(self.output_dir / "history.jsonl", result)
        _atomic_write_json(self.output_dir / "status.json", result)
        print_grid(records, int(config["boardsize"]))
        print(
            f"HEX_EVAL_RESULT cycle={cycle} "
            f"score={summary['model_win_rate']:.6f} "
            f"wins={summary['model_wins']} losses={summary['model_losses']} "
            f"unscored={summary['unscored']}",
            flush=True,
        )
        return training_metrics(summary)


# =============================================================================
# Shared opening enumeration and value diagnostics
# Source: src/nanoalphazero/training.py
# =============================================================================


def _legal_mask_from_state(env_state, config):
    if config["env_id"] == "chess":
        return unpack_bitmask_vmap(env_state.legal_action_bitmask)
    return env_state.legal_action_mask


def all_opening_actions(wenv, config, plies=1):
    """Every legal opening LINE of length `plies` from the initial position.

    Returns a 2-D int array of shape (num_openings, plies); row i is one forced
    move sequence. plies=1 -> one row per legal first move (ttt->9, connect4->7,
    hex4x4->16, chess->20). plies=2 -> every legal (move1, move2) pair (chess
    ->~400), which probes a wider, more balanced slice of the opening tree and
    so gives a lower-variance ladder score. Lines that terminate before `plies`
    moves are dropped (can't host a full game), keeping the result rectangular.
    """
    env_state = wenv.init_dummy_estate(batch_size=1)
    legal = np.asarray(_legal_mask_from_state(env_state, config)[0])
    lines = [[int(a)] for a in np.nonzero(legal)[0]]
    for _ in range(int(plies) - 1):
        n = len(lines)
        env_state = wenv.init_dummy_estate(batch_size=n)
        depth = len(lines[0])
        for d in range(depth):  # replay the line built so far, batched
            col = jnp.asarray([line[d] for line in lines], dtype=jnp.int32)
            env_state = wenv.step(env_state, col)
        legal_masks = np.asarray(_legal_mask_from_state(env_state, config))
        terminated = np.asarray(env_state.terminated)
        next_lines = []
        for i, line in enumerate(lines):
            if terminated[i]:
                continue  # game over before reaching `plies` moves -> drop
            for a in np.nonzero(legal_masks[i])[0]:
                next_lines.append(line + [int(a)])
        lines = next_lines
    return np.asarray(lines, dtype=np.int32)


def _run_ttt_diagnostics(model_ts, wenv, config):
    boardsize = config.get("boardsize", 3)
    env = pgx1.make("tic_tac_toe")
    dummy_state = env.init(jax.random.PRNGKey(0))
    obs = env.observe(dummy_state, dummy_state.current_player)
    obs_b = obs[jnp.newaxis, ...]
    legal_b = dummy_state.legal_action_mask[jnp.newaxis, ...]

    logits, values = model_ts.apply_fn(
        {"params": model_ts.params}, obs_b, legal_b, deterministic=True
    )
    logits = logits.flatten()
    value = float(values.flatten()[0])

    print(f"  Diagnostics: P1 argmax={int(jnp.argmax(logits))}, value={value:.3f}")
    logits_2d = np.array(logits).reshape((boardsize, boardsize))

    print(f"  Logits (empty board), value={value:.3f}:")
    for r in range(boardsize):
        print("    " + "  ".join(f"{logits_2d[r, c]:+.2f}" for c in range(boardsize)))


def _get_hex_perfect_play_values(boardsize: int):

    return perfect_play_values(boardsize)


def _run_hex_diagnostics(model_ts, wenv, config):
    boardsize = config["boardsize"]
    batch_size = boardsize * boardsize

    # One blank board per opening; each plays a distinct first move.
    env_state = wenv.init_dummy_estate(batch_size=batch_size)
    all_moves = jnp.arange(batch_size)
    env_state = wenv.step(env_state, all_moves)
    obs = wenv.observe(env_state, env_state.current_player)

    _logits, values = model_ts.apply_fn(
        {"params": model_ts.params},
        obs,
        env_state.legal_action_mask,
        deterministic=True,
    )
    values_2d = np.array(values.flatten()).reshape((boardsize, boardsize))

    print(
        "\n--- Hex value-head after each Black opening "
        "(value = White-to-move perspective; negative => Black-winning) ---"
    )
    for r in range(boardsize):
        print("  " + " ".join(f"{values_2d[r, c]:+.2f}" for c in range(boardsize)))
    gt = _get_hex_perfect_play_values(boardsize)
    if gt is None:
        print(f"  (no ground-truth perfect-play table for {boardsize}x{boardsize})")
        return

    # B = Black-winning opening (gt -1), . = Black-losing opening (gt +1)
    print("  Perfect play (B=Black wins / .=Black loses), [x]=model sign mismatch:")
    pred_sign = np.sign(values_2d)
    for r in range(boardsize):
        cells = []
        for c in range(boardsize):
            truth = "B" if gt[r, c] < 0 else "."
            mismatch = pred_sign[r, c] != np.sign(gt[r, c])
            cells.append(f"[{truth}]" if mismatch else f" {truth} ")
        print("  " + "".join(cells))

    mse = float(np.mean((values_2d - gt) ** 2))
    sign_acc = float(np.mean(pred_sign == np.sign(gt)))
    print(f"  MSE vs perfect = {mse:.4f} | sign accuracy = {sign_acc:.3f}")


def _get_connect4_perfect_play_values():
    # idx:            0    1    2    3    4    5    6
    vals = np.array([+1.0, +1.0, 0.0, -1.0, 0.0, +1.0, +1.0], dtype=np.float32)
    labels = ["L", "L", "D", "W", "D", "L", "L"]  # outcome for P1
    return vals, labels


def _run_connect4_diagnostics(model_ts, wenv, config):
    num_cols = 7
    # One blank board per opening column; each plays a distinct first move.
    env_state = wenv.init_dummy_estate(batch_size=num_cols)
    all_moves = jnp.arange(num_cols)
    env_state = wenv.step(env_state, all_moves)
    obs = wenv.observe(env_state, env_state.current_player)

    _logits, values = model_ts.apply_fn(
        {"params": model_ts.params},
        obs,
        env_state.legal_action_mask,
        deterministic=True,
    )
    values = np.array(values.flatten())  # (7,) P2-to-move perspective

    gt, labels = _get_connect4_perfect_play_values()

    print(
        "\n--- Connect4 value-head after each opening column "
        "(value = P2-to-move perspective; negative => P1-winning) ---"
    )
    print("  col:    " + "  ".join(f"{c+1:>5d}" for c in range(num_cols)))
    print("  value:  " + "  ".join(f"{values[c]:+.2f}" for c in range(num_cols)))
    print(
        "  perfect:"
        + "  ".join(f"{labels[c]:>5s}" for c in range(num_cols))
        + "    (W=P1 wins, D=draw, L=P1 loses)"
    )

    # Directional correctness: W -> value<0, L -> value>0, D -> |value| small.
    draw_tol = 0.5
    correct = []
    for c in range(num_cols):
        if labels[c] == "W":
            correct.append(values[c] < 0)
        elif labels[c] == "L":
            correct.append(values[c] > 0)
        else:  # draw
            correct.append(abs(values[c]) < draw_tol)
    print("  match:  " + "  ".join("  ok " if ok else "  x  " for ok in correct))

    mse = float(np.mean((values - gt) ** 2))
    acc = float(np.mean(correct))
    print(f"  MSE vs perfect = {mse:.4f} | directional accuracy = {acc:.3f}")


def _run_go_diagnostics(model_ts, wenv, config):
    """Go diagnostic: policy logits on the empty board and the value head after
    each non-pass opening move, both laid out as boardsize x boardsize grids.

    The pass action (index boardsize**2) is excluded from both grids; its policy
    logit is printed inline next to the empty-board value for reference.
    """
    boardsize = config["boardsize"]
    n = boardsize * boardsize  # number of board points (pass is action index n)

    # --- Empty-board policy logits (non-pass actions) + value ---
    env_state = wenv.init_dummy_estate(batch_size=1)
    obs = wenv.observe(env_state, env_state.current_player)
    logits, values = model_ts.apply_fn(
        {"params": model_ts.params},
        obs,
        env_state.legal_action_mask,
        deterministic=True,
    )
    logits = np.array(logits.flatten())
    pass_logit = float(logits[n])
    value0 = float(values.flatten()[0])
    logits_2d = logits[:n].reshape((boardsize, boardsize))

    print(
        f"\n--- Go policy logits on empty board "
        f"(value={value0:+.3f}, pass logit={pass_logit:+.2f}) ---"
    )
    for r in range(boardsize):
        print("  " + " ".join(f"{logits_2d[r, c]:+.2f}" for c in range(boardsize)))

    # --- Value head after each non-pass opening move ---
    # One blank board per opening point; each plays a distinct first move.
    env_state = wenv.init_dummy_estate(batch_size=n)
    all_moves = jnp.arange(n)  # 0..n-1: every board point (excludes pass=n)
    env_state = wenv.step(env_state, all_moves)
    obs = wenv.observe(env_state, env_state.current_player)
    _logits, values = model_ts.apply_fn(
        {"params": model_ts.params},
        obs,
        env_state.legal_action_mask,
        deterministic=True,
    )
    values_2d = np.array(values.flatten()).reshape((boardsize, boardsize))

    print(
        "\n--- Go value head after each opening move "
        "(value = opponent-to-move perspective; negative => opening side winning) ---"
    )
    for r in range(boardsize):
        print("  " + " ".join(f"{values_2d[r, c]:+.2f}" for c in range(boardsize)))


# =============================================================================
# Vector MuZero: representation h, dynamics g, prediction f
# Source: src/nanoalphazero/research/muzero/model.py
# h maps each root observation to [batch, width]. g concatenates the latent
# with a one-hot action and predicts an immediate reward and next latent.
# f predicts policy logits and side-to-move value. The three towers have
# separate parameters; their latent interface is trained end to end.
# =============================================================================

"""Small vector-latent MuZero baseline, with no observation reconstruction."""


def scale_gradient(x, scale):
    return scale * x + (1 - scale) * jax.lax.stop_gradient(x)


def normalize(x):
    lo = jnp.min(x, axis=-1, keepdims=True)
    span = jnp.max(x, axis=-1, keepdims=True) - lo
    return (x - lo) / jnp.maximum(span, 1e-5)


def build_model(config):
    if config.get("network", "vector") == "vector":
        return MuZero(config["num_actions"], config["width"], config["depth"])
    if config["network"] == "spatial":
        return SpatialMuZero(config["num_actions"], config["env_id"], config["width"], config["depth"],
                             config["activation"], config["use_rvgl"],
                             remat_blocks=config.get("remat_blocks", False))
    raise ValueError("Unknown MuZero network")


class Tower(nn.Module):
    width: int
    depth: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.width)(x))
        for _ in range(self.depth):
            residual = nn.relu(nn.Dense(self.width)(x))
            x = nn.relu(x + nn.Dense(self.width)(residual))
        return x


class MuZero(nn.Module):
    num_actions: int
    width: int = 128
    depth: int = 2

    def setup(self):
        self.representation = Tower(self.width, self.depth)
        self.dynamics = Tower(self.width, self.depth)
        self.prediction = Tower(self.width, self.depth)
        self.policy = nn.Dense(self.num_actions)
        self.value = nn.Dense(1)
        self.reward = nn.Dense(1)

    def predict(self, latent):
        features = self.prediction(latent)
        return self.policy(features), jnp.tanh(self.value(features)[..., 0])

    def initial(self, observation):
        flat = observation.astype(jnp.float32).reshape((observation.shape[0], -1))
        latent = normalize(self.representation(flat))
        logits, value = self.predict(latent)
        return latent, logits, value

    def recurrent(self, latent, action):
        action = jax.nn.one_hot(action, self.num_actions)
        features = self.dynamics(jnp.concatenate([latent, action], axis=-1))
        reward = jnp.tanh(self.reward(features)[..., 0])
        latent = normalize(features)
        logits, value = self.predict(latent)
        return latent, reward, logits, value

    def __call__(self, observation, action):
        latent, _, _ = self.initial(observation)
        return self.recurrent(latent, action)


# =============================================================================
# Optional activation recomputation to reduce training memory
# Source: src/nanoalphazero/research/muzero/remat.py
# =============================================================================

"""Memory adapter composing the production trunk's unchanged building blocks."""


class RematerializedTrunk(KataGoTrunk):
    """Recompute each nested block during backward; preserve parameter paths.

    This mirrors only KataGoTrunk's composition because its block factory is
    not configurable. All layers, initializers and blocks remain production
    implementations. Explicit block names retain checkpoint/init equivalence.
    """

    @nn.compact
    def __call__(self, input_spatial, input_global=None, mask=None):
        input_spatial = input_spatial.astype(jnp.float32)
        if mask is None:
            mask = jnp.ones_like(input_spatial[..., :1])
        mask_sum = jnp.sum(mask, axis=(1, 2))
        out = nn.Conv(self.c_trunk, (3, 3), use_bias=False,
                      kernel_init=kata_init(0.8, self.activation))(input_spatial * mask)
        if input_global is not None:
            out = out + nn.Dense(self.c_trunk, use_bias=False,
                                 kernel_init=kata_init(0.6, self.activation))(
                                     input_global.astype(jnp.float32))[:, None, None, :]
        block = nn.remat(NestedBottleneckResBlock)
        fixup_scale = 1.0 / math.sqrt(len(self.block_gpool))
        for index, use_gpool in enumerate(self.block_gpool):
            out = out + block(
                self.c_trunk, self.c_mid, self.internal_length,
                fixup_scale, self.activation,
                c_gpool=self.c_gpool if use_gpool else None,
                use_rvgl=self.use_rvgl,
                name=f"NestedBottleneckResBlock_{index}",
            )(out, mask, mask_sum)
        out = _ACTS[self.activation](NormMask()(out, mask))
        return out, mask, mask_sum


# =============================================================================
# Spatial MuZero: board-shaped latent and action planes
# Source: src/nanoalphazero/research/muzero/spatial.py
# The latent is [batch, board_height, board_width, channels]. Its cells
# are learned features, not reconstructed stones or simulator states.
# Action planes encode action coordinates/type, never future legality.
# =============================================================================

"""MuZero with production KataGo trunk/heads and a spatial latent state."""


def normalize_spatial(x):
    lo = jnp.min(x, axis=(1, 2, 3), keepdims=True)
    span = jnp.max(x, axis=(1, 2, 3), keepdims=True) - lo
    return (x - lo) / jnp.maximum(span, 1e-5)


def action_planes(action, height, width, num_actions, env_id):
    """Encode the action definition, without a board or legality oracle."""
    batch = action.shape[0]
    if env_id == "connect_four":
        return jnp.broadcast_to(jax.nn.one_hot(action, width)[:, None, :, None],
                                (batch, height, width, 1))
    if env_id == "chess":
        planes = jax.nn.one_hot(action, num_actions).reshape(batch, height, width, 73)
        # Inverse of production ChessPolicyHead's final coordinate rotation.
        return jnp.rot90(planes, k=1, axes=(1, 2))
    board = jax.nn.one_hot(action, height * width).reshape(batch, height, width, 1)
    if env_id.startswith("go_"):
        passing = jnp.broadcast_to((action == height * width)[:, None, None, None], board.shape)
        return jnp.concatenate([board, passing.astype(board.dtype)], -1)
    if num_actions != height * width:
        raise ValueError("Unknown spatial action encoding")
    return board


class Prediction(nn.Module):
    num_actions: int
    env_id: str
    width: int
    depth: int
    activation: str

    @nn.compact
    def __call__(self, latent, return_wdl=False):
        cfg = resolve_preset(f"b{self.depth}c{self.width}nbt")
        mask = jnp.ones_like(latent[..., :1])
        mask_sum = jnp.sum(mask, axis=(1, 2))
        if self.env_id.startswith("go_"):
            head = GoPolicyHead(cfg["c_p1"], cfg["c_g1"], self.activation)
        elif self.env_id == "chess":
            head = ChessPolicyHead(cfg["c_p1"], cfg["c_g1"], self.activation)
        else:
            head = GenericPolicyHead(self.num_actions, cfg["c_p1"], self.activation)
        logits = head(latent, mask, mask_sum)
        value_logits = ValueHead(cfg["c_v1"], cfg["c_v2"], self.activation)(latent, mask, mask_sum)
        if return_wdl:
            return logits, value_from_logits(value_logits), jax.nn.softmax(value_logits, -1)
        return logits, value_from_logits(value_logits)


class SpatialMuZero(nn.Module):
    num_actions: int
    env_id: str
    width: int
    depth: int
    activation: str = "mish"
    use_rvgl: bool = True
    remat_blocks: bool = False

    def setup(self):
        cfg = resolve_preset(f"b{self.depth}c{self.width}nbt")
        trunk = {k: v for k, v in cfg.items() if k in (
            "c_trunk", "c_mid", "c_gpool", "block_gpool", "internal_length")}
        trunk_type = RematerializedTrunk if self.remat_blocks else KataGoTrunk
        self.representation = trunk_type(**trunk, activation=self.activation, use_rvgl=self.use_rvgl)
        self.dynamics = trunk_type(**trunk, activation=self.activation, use_rvgl=self.use_rvgl)
        self.prediction = Prediction(self.num_actions, self.env_id, self.width, self.depth, self.activation)
        self.reward = ValueHead(cfg["c_v1"], cfg["c_v2"], self.activation)

    def initial(self, observation):
        latent, _, _ = self.representation(observation)
        latent = normalize_spatial(latent)
        logits, value = self.prediction(latent)
        return latent, logits, value

    def recurrent(self, latent, action):
        planes = action_planes(action, latent.shape[1], latent.shape[2], self.num_actions, self.env_id)
        features, mask, mask_sum = self.dynamics(jnp.concatenate([latent, planes], -1))
        reward = value_from_logits(self.reward(features, mask, mask_sum))
        latent = normalize_spatial(features)
        logits, value = self.prediction(latent)
        return latent, reward, logits, value

    def initial_with_wdl(self, observation):
        latent, _, _ = self.representation(observation)
        latent = normalize_spatial(latent)
        logits, value, probabilities = self.prediction(latent, return_wdl=True)
        return latent, logits, value, probabilities

    def __call__(self, observation, action):
        latent, _, _ = self.initial(observation)
        return self.recurrent(latent, action)


# =============================================================================
# Latent search: root legality, learned hypothetical transitions
# Source: src/nanoalphazero/research/muzero/search.py
# Encode the root once. Both search rungs expand through g and f only.
# The simulator supplies root legality. Internal nodes use the full action
# vocabulary. Alternating-player backups are reward - discount * value.
# =============================================================================

"""Adapters for the unchanged production two-rung search."""


@chex.dataclass(frozen=True)
class ResearchPolicyOutput(PolicyOutput):
    raw_policy_logits: object = None
    predicted_value: object = None


def legal_actions(state):
    if hasattr(state, "legal_action_bitmask"):
        return unpack_bitmask_vmap(state.legal_action_bitmask)
    return state.legal_action_mask


def latent_search(model, params, observation, legal, key, *, roots, survivors,
                  gumbel_scale=1.0, discount=1.0, alternating=True,
                  root_temperature=1., value_scale=1., maxvisit_init=50., rescale_values=False,
                  diagnostics=False):
    """No environment object or true future state enters this function."""
    latent, logits, value = model.apply({"params": params}, observation, method=model.initial)

    def recurrent(params, keys, action, latent):
        latent, reward, logits, value = model.apply(
            {"params": params}, latent, action, method=model.recurrent
        )
        signed_discount = -discount if alternating else discount
        return RecurrentFnOutput(
            reward=reward, discount=jnp.full_like(value, signed_discount),
            prior_logits=logits, value=value,
        ), latent

    output = gumbel_muzero_policy_1sh(
        params, key, RootFnOutput(prior_logits=logits / root_temperature, value=value, embedding=latent),
        recurrent, num_root_considered=roots, num_survivors=survivors,
        invalid_actions=~legal, gumbel_scale=gumbel_scale,
        value_scale=value_scale, maxvisit_init=maxvisit_init, rescale_values=rescale_values,
    )
    if diagnostics:
        return ResearchPolicyOutput(**vars(output), raw_policy_logits=logits, predicted_value=value)
    return output


def make_search(model, env, config, policy_only=False):
    @functools.partial(jax.jit, static_argnums=(4, 5))
    def search(key, state, params, gumbel_scale, batch_size, num_simulations=None):
        obs = env.observe(state, state.current_player)
        legal = legal_actions(state)
        if policy_only:
            _, logits, _ = model.apply({"params": params}, obs, method=model.initial)
            logits = jnp.where(legal, logits, jnp.finfo(logits.dtype).min)
            return PolicyOutput(action=jnp.argmax(logits, -1),
                                action_weights=jax.nn.softmax(logits))
        return latent_search(
            model, params, obs, legal, key, roots=config["roots"],
            survivors=config["survivors"], gumbel_scale=gumbel_scale,
            discount=config["discount"],
            root_temperature=config.get("root_temperature", 1.),
            value_scale=config.get("value_scale", 1.), maxvisit_init=config.get("maxvisit_init", 50.),
            rescale_values=config.get("rescale_values", False),
            diagnostics=True,
        )
    return search


# =============================================================================
# Contiguous replay, signed returns, absorbing and padded targets
# Source: src/nanoalphazero/research/muzero/replay.py
# At unroll index k: policy/value describe s_k; reward describes action
# a_k taking s_k to s_(k+1). Terminal absorbing targets differ from padding
# beyond a truncated boundary; masks keep those distinctions in the loss.
# =============================================================================

"""Episode replay and aligned unrolls, separate from AlphaZero outcome records."""


def discounted_returns(rewards, discounts, bootstrap):
    """[B,T] actor rewards; signed discounts map next-player values to actors."""
    def backup(value, transition):
        reward, discount = transition
        value = reward + discount * value
        return value, value
    _, values = jax.lax.scan(backup, bootstrap, (rewards.T, discounts.T), reverse=True)
    return values.T


def make_replay(config):
    # A Flashbax time slot contains an entire episode. Sampling cannot cross
    # games, circular-buffer wraparound, or independently reset batch lanes.
    return flashbax.make_trajectory_buffer(
        add_batch_size=config["selfplay_batch_size"],
        sample_batch_size=config["train_batch_size"],
        sample_sequence_length=1, period=1, min_length_time_axis=1,
        max_length_time_axis=config["replay_batches"],
    )


def sequences(episodes, key, unroll, starts=None):
    """Uniform game then uniform real position; pad terminal tails only.

    reward[:, k] trains g(s[k], a[k]); policy/value[:, k] train f(s[k]).
    Truncated tails have a bootstrap value at their boundary, but no invented
    rewards/policies beyond it. Terminal tails train absorbing zero values and
    rewards with random actions and no policy loss.
    """
    batch, length = episodes["action"].shape
    key, action_key = jax.random.split(key)
    if starts is None:
        starts = jax.random.randint(key, (batch,), 0, episodes["length"])
    indices = starts[:, None] + jnp.arange(unroll + 1)
    rows = jnp.arange(batch)[:, None]
    clipped = jnp.minimum(indices, length - 1)
    real = indices < episodes["length"][:, None]
    boundary = indices == episodes["length"][:, None]
    terminal = episodes["terminal"][:, None]
    value = jnp.where(real, episodes["value"][rows, clipped], 0.0)
    value = jnp.where(boundary & ~terminal, episodes["bootstrap"][:, None], value)
    random_actions = jax.random.randint(
        action_key, (batch, unroll), 0, episodes["policy"].shape[-1]
    )
    result = {
        "observation": episodes["observation"][jnp.arange(batch), starts],
        "action": jnp.where(real[:, :-1], episodes["action"][rows, clipped][:, :-1], random_actions),
        "reward": jnp.where(real[:, :-1], episodes["reward"][rows, clipped][:, :-1], 0.0),
        "policy": episodes["policy"][rows, clipped],
        "value": value,
        "policy_mask": real,
        "value_mask": real | boundary | terminal,
        "reward_mask": real[:, :-1] | terminal,
    }
    result["sample_info"] = {
        "ep_step": starts, "ep_termination_step": episodes["length"] - 1,
        "terminal": episodes["terminal"], "action": episodes["action"][jnp.arange(batch), starts],
        "is_from_selfplay": jnp.ones(batch, bool), "is_pending_reward_i8": jnp.zeros(batch, jnp.int8),
        "is_fresh_i8": jnp.ones(batch, jnp.int8),
        "is_exploration": episodes.get("exploration", jnp.zeros((batch, length), bool))[jnp.arange(batch), starts],
        "game_id": episodes.get("game_id", jnp.zeros(batch, jnp.uint32)),
        "row_id": episodes.get("row_id", jnp.arange(batch, dtype=jnp.int32) + 1),
    }
    return result


# =============================================================================
# Self-play staging -> consume/drain -> sequence replay
# Source: src/nanoalphazero/research/muzero/staging.py
# =============================================================================

"""Completed-episode staging using AlphaZero's actual fresh-position consumer.

Collection/backfill remains synchronous: no unresolved outcome enters staging.
Only consumption selects positions; Flashbax replay stores materialized unrolls.
"""


@chex.dataclass(frozen=True)
class ConsumptionView:
    global_step_id: jax.Array
    is_fresh_i8: jax.Array
    is_valid_sample: jax.Array
    lane: jax.Array
    slot: jax.Array
    start: jax.Array


@struct.dataclass
class StagingState:
    episodes: object
    fresh: jax.Array
    generation: jax.Array


def make_staging(config):
    batch = config["selfplay_batch_size"]
    slots = config["staging_batches"]
    length = config["max_steps"]
    consume_size = config["consume_size"]
    storage = flashbax.make_trajectory_buffer(
        add_batch_size=batch, sample_batch_size=1, sample_sequence_length=1,
        period=1, min_length_time_axis=1, max_length_time_axis=slots)
    # The production consumer needs only these six scalar fields. Its unused
    # initial buffer is tiny; real payload stays in the episode storage above.
    dummy = ConsumptionView(global_step_id=jnp.uint32(0), is_fresh_i8=jnp.int8(0),
                            is_valid_sample=jnp.bool_(False), lane=jnp.int32(0),
                            slot=jnp.int32(0), start=jnp.int32(0))
    production, _ = make_selfplay_buffer(dict(
        enable_sharding=False, selfplay_buffer_add_batch_size=1,
        selfplay_buffer_sample_batch_size=1, selfplay_buffer_min_len=1,
        selfplay_buffer_max_len=1, selfplay_buffer_consume_size=consume_size), dummy)

    def init(example):
        return StagingState(vars(storage.init(example)), jnp.zeros((batch, slots, length), bool),
                            jnp.uint32(0))

    def add_backfill(state, episodes):
        # make_collect completed the signed return scan before handing us the
        # batch. Real prefix validity must agree with its explicit length.
        real = jnp.arange(length)[None] < episodes["length"][:, None]
        malformed = jnp.any(real != episodes["valid"]) | jnp.any(episodes["length"] <= 0)
        index = state.episodes["current_index"]
        overwritten = jnp.sum(state.fresh[:, index])
        eligible = real & ~episodes["exploration"]
        if "game_id" in episodes:
            episodes = {**episodes, "game_id": state.generation * batch + jnp.arange(batch, dtype=jnp.uint32) + 1}
        updated = storage.add(TrajectoryBufferState(**state.episodes), jax.tree.map(lambda x: x[:, None], episodes))
        return state.replace(episodes=vars(updated), fresh=state.fresh.at[:, index].set(eligible),
                             generation=state.generation + 1), {
            "staging/overwritten_fresh": overwritten,
            "anomalies/selfplay_buffer/eviction_n_is_from_selfplay": jnp.sum(state.episodes["experience"]["length"][:, index]),
            "staging/malformed_episodes": malformed.astype(jnp.int32),
            "staging/new_eligible": jnp.sum(eligible),
            "staging/exploration_excluded": jnp.sum(real & episodes["exploration"]),
        }

    def consume(state, key):
        # Use the exact production randomized top-k/fresh-clearing logic.
        # A slot contains a complete episode; slot+start identify its unroll.
        shape = (batch, slots, length)
        flatten = lambda x: jnp.broadcast_to(x, shape).reshape(batch, slots * length)
        valid = state.fresh.reshape(batch, slots * length)
        view = ConsumptionView(
            global_step_id=jnp.full(valid.shape, state.generation, jnp.uint32),
            is_fresh_i8=valid.astype(jnp.int8), is_valid_sample=valid,
            lane=flatten(jnp.arange(batch)[:, None, None]),
            slot=flatten(jnp.arange(slots)[None, :, None]),
            start=flatten(jnp.arange(length)[None, None, :]))
        production_state = CustomTrajectoryBufferState(
            experience=view, current_index=jnp.int32(0), is_full=jnp.bool_(True),
            num_valid_consumable=jnp.sum(valid))
        after, selected, _ = production.consume(production_state)
        selected = jax.tree.map(lambda x: x[:, 0], selected)
        episodes = jax.tree.map(lambda x: x[selected.lane, selected.slot], state.episodes["experience"])
        unroll = sequences(episodes, key, config["unroll"], starts=selected.start)
        unroll["sample_info"].update(row_id=selected.lane + 1, col_id=selected.slot * length + selected.start,
                                     is_fresh_i8=selected.is_fresh_i8,
                                     is_from_selfplay=selected.is_valid_sample)
        # Defensive masking if called with fewer than consume_size positions.
        # The drain loop below never inserts these partial batches into replay.
        unroll = {**unroll, **{name: unroll[name] & selected.is_valid_sample[:, None]
                              for name in ("policy_mask", "value_mask", "reward_mask")}}
        return state.replace(fresh=after.experience.is_valid_sample.reshape(shape)), unroll

    replay = flashbax.make_trajectory_buffer(
        add_batch_size=consume_size, sample_batch_size=config["train_batch_size"],
        sample_sequence_length=1, period=1, min_length_time_axis=1,
        max_length_time_axis=max(1, config["replay_positions"] // consume_size))

    def drain(state, replay_state, key):
        count = jnp.sum(state.fresh) // consume_size

        def consumption_metrics(unroll, before, after):
            valid = unroll["policy_mask"][:, 0]
            info = unroll["sample_info"]
            prefix = "anomalies/selfplay_buffer-consume/"
            result = {
                prefix + "n_is_valid_but_stale": jnp.sum(valid & (info["is_fresh_i8"] == 0)),
                prefix + "n_is_valid_but_not_real": jnp.sum(valid & ~info["is_from_selfplay"]),
                prefix + "n_is_valid_but_is_pending_reward_i8": jnp.sum(valid & (info["is_pending_reward_i8"] == 1)),
                prefix + "n_is_valid_but_is_exploration": jnp.sum(valid & info["is_exploration"]),
                prefix + "n_not_is_valid_": jnp.sum(~valid),
                prefix + "n_is_valid_but_game_id_zero": jnp.sum(valid & (info["game_id"] == 0)),
            }
            position_ids = jnp.sort(jnp.where(valid, info["row_id"] * slots * length + info["col_id"], -1))
            result[prefix + "n_dupes"] = jnp.sum((position_ids[1:] == position_ids[:-1]) & (position_ids[1:] >= 0))
            ids = jnp.sort(jnp.where(valid, info["game_id"], 0))
            result["selfplay_buffer-consume/n_unique_game_ids"] = jnp.sum((ids != jnp.roll(ids, 1)) & (ids > 0)) + ((ids[0] == ids[-1]) & (ids[0] > 0))
            for name, field in (("term_step", "ep_termination_step"), ("ep_step", "ep_step")):
                result[f"selfplay_buffer-comp/max_{name}"] = jnp.max(jnp.where(valid, info[field], -1))
                result[f"selfplay_buffer-comp/min_{name}"] = jnp.min(jnp.where(valid, info[field], length))
                result[f"selfplay_buffer-comp/avg_{name}"] = jnp.sum(jnp.where(valid, info[field], 0)) / jnp.maximum(jnp.sum(valid), 1)
            for name, values in (("n_is_from_selfplay", info["is_from_selfplay"]), ("n_is_valid", valid),
                                  ("n_is_pending_reward_i8", info["is_pending_reward_i8"]), ("n_is_exploration", info["is_exploration"])):
                result[f"selfplay_buffer-comp/{name}"] = jnp.sum(values)
            after_real = jnp.arange(length)[None, None] < after.episodes["experience"]["length"][..., None]
            result.update({
                "selfplay_buffer-consume/n_consumables": jnp.sum(before.fresh),
                "selfplay_buffer-consume/returnable_n_all_valid_completed": jnp.sum(before.fresh),
                "selfplay_buffer-consume/returnable_n_fresh_returnable_mask": jnp.sum(after.fresh),
                "selfplay_buffer-consume/after_return_n_overall_fresh": jnp.sum(after.fresh),
                "selfplay_buffer-consume/after_return_n_overall_fresh_valid": jnp.sum(after.fresh & after_real),
                "selfplay_buffer-consume/after_return_n_overall_fresh_not_from_selfplay": jnp.sum(after.fresh & ~after_real),
                "selfplay_buffer-consume/after_return_n_overall_fresh_from_selfplay": jnp.sum(after.fresh & after_real),
                "selfplay_buffer-consume/after_return_n_overall_fresh_exploration": jnp.sum(after.fresh & after.episodes["experience"]["exploration"]),
            })
            return result

        # Infer scalar shapes without executing a consume or clearing freshness.
        dummy = jax.eval_shape(consume, state, key)[1]
        zero_batch = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), dummy)
        initial_metrics = jax.tree.map(jnp.zeros_like, consumption_metrics(zero_batch, state, state))

        def one(_, carry):
            staged, replayed, rng, _ = carry
            rng, subkey = jax.random.split(rng)
            before = staged
            staged, unroll = consume(staged, subkey)
            replayed = replay.add(replayed, jax.tree.map(lambda x: x[:, None], unroll))
            return staged, replayed, rng, consumption_metrics(unroll, before, staged)

        state, replay_state, key, consume_metrics = jax.lax.fori_loop(0, count, one, (state, replay_state, key, initial_metrics))
        return state, replay_state, key, {
            **consume_metrics,
            "drain/num_valid_consumable": count * consume_size + jnp.sum(state.fresh),
            "drain/n_slices": count, "drain/inserted_positions": count * consume_size,
            "drain/remaining_fresh": jnp.sum(state.fresh)}

    return init, add_backfill, consume, drain, replay


# =============================================================================
# Actual self-play and differentiable recurrent training
# Source: src/nanoalphazero/research/muzero/learning.py
# The collector may step the real environment; hypothetical search may not.
# Rewards use the acting player's perspective; signed discounts propagate
# returns between players. Recurrent gradients are scaled by 0.5, while
# forward latent values are unchanged. Unroll length is independent of
# network depth and the fixed two-rung search expansion budget.
# =============================================================================

"""Self-play episode generation and end-to-end unrolled optimization."""


def make_collect(env, model, search, config):
    batch = config["selfplay_batch_size"]

    def collect(params, key):
        key, init_key = jax.random.split(key)
        state = env.init(jax.random.split(init_key, batch))
        if config.get("exploration_mode", "fixed") == "random_switch":
            key, exploration_key = jax.random.split(key)
            switch = jax.random.randint(exploration_key, (batch,), 0,
                                        max(config["exploration_moves"], 1))
        else:
            switch = config["exploration_moves"]

        def step(carry, t):
            state, ended, key = carry
            key, sk, ek, ak = jax.random.split(key, 4)
            obs = env.observe(state, state.current_player)
            policy = search(sk, state, params, 1.0, batch)
            # Preserve production's visit-count exploration convention.
            logits = jnp.log(jnp.maximum(policy.visit_counts, 1e-30))
            logits = jnp.where(legal_actions(state), logits, -1e30)
            sampled = jax.random.categorical(ak, logits)
            action = jnp.where(t < switch, sampled, policy.action).astype(jnp.int32)
            next_state = env.step(state, action, jax.random.split(ek, batch))
            reward = next_state.rewards[jnp.arange(batch), state.current_player]
            reward = jnp.where(ended, 0.0, reward)
            sign = jnp.where(state.current_player == next_state.current_player, 1.0, -1.0)
            discount = jnp.where(ended | next_state.terminated, 0.0, sign * config["discount"])
            row = {"observation": obs, "action": action,
                   "player": state.current_player,
                   "next_legal_count": jnp.sum(legal_actions(next_state), axis=-1),
                   "raw_policy_logits": getattr(policy, "raw_policy_logits", jnp.log(jnp.maximum(policy.action_weights, 1e-30))),
                   "predicted_value": getattr(policy, "predicted_value", jnp.zeros(batch)),
                   "reward": reward, "discount": discount,
                   "policy": policy.action_weights, "valid": ~ended,
                   "exploration": (~ended) & (t < switch),
                   "illegal": ~ended & ~legal_actions(state)[jnp.arange(batch), action]}
            ended = ended | next_state.terminated | next_state.truncated
            # Freeze completed states, so truncation bootstrap remains at the
            # actual boundary and terminal rewards cannot leak into padding.
            next_state = jax.tree.map(
                lambda old, new: jnp.where(carry[1].reshape((batch,) + (1,) * (new.ndim - 1)), old, new),
                state, next_state,
            )
            return (next_state, ended, key), row

        (state, _, _), rows = jax.lax.scan(
            step, (state, jnp.zeros(batch, bool), key), jnp.arange(config["max_steps"])
        )
        rows = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), rows)
        _, _, bootstrap = model.apply(
            {"params": params}, env.observe(state, state.current_player), method=model.initial
        )
        bootstrap = jnp.where(state.terminated, 0.0, bootstrap)
        # Padded rows must pass the boundary bootstrap backwards unchanged.
        discounts = jnp.where(rows["valid"], rows["discount"], 1.0)
        values = discounted_returns(rows["reward"], discounts, bootstrap)
        return {**rows, "value": values, "bootstrap": bootstrap,
                "length": jnp.sum(rows["valid"], axis=1), "terminal": state.terminated,
                "final_rewards": state.rewards, "game_id": jnp.arange(batch, dtype=jnp.uint32) + 1,
                "row_id": jnp.arange(batch, dtype=jnp.int32) + 1}
    return collect


def unroll_predictions(model, params, batch, with_wdl=False, remat=False):
    wdl = None
    include_wdl = with_wdl and hasattr(model, "initial_with_wdl")

    def initial(p, observation):
        return model.apply({"params": p}, observation,
                           method=model.initial_with_wdl if include_wdl else model.initial)

    def recurrent(p, latent, action):
        return model.apply({"params": p}, latent, action, method=model.recurrent)

    # This recomputes activations during reverse-mode differentiation. It
    # changes neither the model nor the logical optimizer batch, and writes
    # no files. Keep the eager baseline available for numerical comparisons.
    if remat:
        initial = jax.checkpoint(initial)
        recurrent = jax.checkpoint(recurrent)
    if include_wdl:
        latent, logits, value, wdl = initial(params, batch["observation"])
    else:
        latent, logits, value = initial(params, batch["observation"])
    policies, values, rewards = [logits], [value], []
    for k in range(batch["action"].shape[1]):
        latent, reward, logits, value = recurrent(
            params, scale_gradient(latent, 0.5), batch["action"][:, k])
        policies.append(logits)
        values.append(value)
        rewards.append(reward)
    policies = jnp.stack(policies, axis=1)
    values = jnp.stack(values, axis=1)
    rewards = jnp.stack(rewards, axis=1)
    return (policies, values, rewards, wdl) if with_wdl else (policies, values, rewards)


def policy_kl(logits, target):
    """KL(search target || prediction), stable for zero target probabilities.

    Replay targets are fixed. Subtracting their entropy from cross-entropy
    changes the reported objective but not its parameter gradient.
    """
    target = jax.lax.stop_gradient(target)
    return optax.softmax_cross_entropy(logits, target) - jnp.sum(
        jax.scipy.special.entr(target), axis=-1)


def loss(model, params, batch, remat=False):
    policies, values, rewards, wdl = unroll_predictions(model, params, batch, with_wdl=True, remat=remat)

    def mean_masked(x, mask):
        return jnp.sum(jnp.where(mask, x, 0.0)) / jnp.maximum(jnp.sum(mask), 1)

    per_state_kl = policy_kl(policies, batch["policy"])
    pi = mean_masked(per_state_kl, batch["policy_mask"])
    v = mean_masked((values - batch["value"]) ** 2, batch["value_mask"])
    r = mean_masked((rewards - batch["reward"]) ** 2, batch["reward_mask"])
    root_mask = batch["policy_mask"][:, 0]
    target_entropy = mean_masked(jnp.sum(jax.scipy.special.entr(batch["policy"][:, 0]), -1), root_mask)
    network_entropy = mean_masked(jnp.sum(jax.scipy.special.entr(jax.nn.softmax(policies[:, 0])), -1), root_mask)
    extra = {"entropy/target_policy": target_entropy, "entropy/network_policy": network_entropy,
             "entropy/diff_network_minus_search": network_entropy - target_entropy,
             "muzero/root_policy_kl": mean_masked(per_state_kl[:, 0], root_mask)}
    if wdl is not None:
        for i, name in enumerate(("win", "loss", "draw")):
            extra[f"wdl/p_{name}_mean"] = mean_masked(wdl[:, i], root_mask)
    if "sample_info" in batch:
        info = batch["sample_info"]
        for step in range(10):
            mask = root_mask & info["terminal"] & (info["ep_termination_step"] - info["ep_step"] == step)
            suffix = f"term_step_{step:02d}"
            extra[f"loss_per_step_PI/pi_{suffix}"] = mean_masked(
                per_state_kl[:, 0], mask)
            extra[f"loss_per_step_V/v_{suffix}"] = mean_masked((values[:, 0] - batch["value"][:, 0]) ** 2, mask)
            extra[f"loss_per_step_count/count_{suffix}"] = jnp.sum(mask)
    return pi + v + r, {"policy_loss": pi, "value_loss": v, "reward_loss": r, **extra}


def train_step(model, state, batch, remat=False):
    (total, metrics), grads = jax.value_and_grad(loss, argnums=1, has_aux=True)(model, state.params, batch, remat)
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    updated = state.replace(step=state.step + 1, params=optax.apply_updates(state.params, updates), opt_state=opt_state)
    grad_norm, param_norm, update_norm = map(optax.global_norm, (grads, state.params, updates))
    dot = sum(jnp.vdot(g, u) for g, u in zip(jax.tree.leaves(grads), jax.tree.leaves(updates)))
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(updates)
    head_metrics = {}
    for name in ("policy", "value"):
        leaves = [v for path, v in flat.items() if name in "/".join(path).lower() and path[0] != "reward"]
        if leaves:
            head_metrics[f"norms/{name}_head_update_ratio"] = optax.global_norm(leaves) / jnp.maximum(param_norm, 1e-12)
    return updated, {
        **metrics, "loss": total, "grad_norm": grad_norm,
        "norms/param_norm": param_norm, "norms/update_norm": update_norm,
        "norms/update_ratio": update_norm / jnp.maximum(param_norm, 1e-12),
        "norms/update_grad_cosine": dot / (grad_norm * update_norm + 1e-12),
        **head_metrics,
    }


# =============================================================================
# MuZero checkpoints: parameters, optimizer, replay, RNG and metadata
# Source: src/nanoalphazero/research/muzero/checkpoint.py
# =============================================================================

"""Versioned MuZero snapshots including optimizer, replay, RNG and full config."""


def metadata(path):
    with safe_open(str(path), framework="flax") as reader:
        meta = reader.metadata()
    if meta.get("format") != "nanoalphazero.research.muzero.v1":
        raise ValueError("Not a MuZero v1 checkpoint")
    return json.loads(meta["config"])


def save(path, state, config):
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("Expected .safetensors checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        flat = flatten_dict(serialization.to_state_dict(state), keep_empty_nodes=True)
        tensors, empty = {}, []
        for keys, value in flat.items():
            name = "/".join(_encode_path_segment(k) for k in keys)
            if value is empty_node:
                empty.append(name)
            else:
                tensors[name] = np.array(jax.device_get(value), copy=True, order="C")
        save_file(tensors, temporary,
                  metadata={"format": "nanoalphazero.research.muzero.v1",
                            "empty_nodes": json.dumps(empty),
                            "config": json.dumps(config, sort_keys=True, allow_nan=False)})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path, template):
    config = metadata(path)
    with safe_open(str(path), framework="flax") as reader:
        flat = {name: reader.get_tensor(name) for name in reader.keys()}
        flat.update({name: empty_node for name in json.loads(reader.metadata()["empty_nodes"])})
    restored = unflatten_dict({tuple(_decode_path_segment(k) for k in name.split("/")): value
                               for name, value in flat.items()})
    return serialization.from_state_dict(template, restored), config


def load_params(path):
    """Load inference parameters without allocating the saved replay buffer."""
    config = metadata(path)
    with safe_open(str(path), framework="flax") as reader:
        flat = {tuple(_decode_path_segment(k) for k in name.split("/")[2:]): reader.get_tensor(name)
                for name in reader.keys() if name.startswith("train/params/")}
    if not flat:
        raise ValueError("Checkpoint has no training parameters")
    return unflatten_dict(flat), config


# =============================================================================
# Optional W&B charts
# Source: src/nanoalphazero/research/muzero/charts.py
# =============================================================================

"""Portable adaptation of az/diagnostics/opening_value_charts.py board helpers.

Preserves AZ's table columns, palette, coordinates, highlights and square cells.
Plotly serializes numeric chart data; no raster images are generated or uploaded.
"""


OPENING_VALUE_COLUMNS = [
    "game", "boardsize", "action", "row", "col", "value", "ground_truth", "highlight", "label",
]


def board_table(values, ground_truth=None, highlights=()):
    values = np.asarray(values)
    size = values.shape[0]
    highlights = set(map(int, highlights))
    rows = [["hex", size, r * size + c, r, c, float(values[r, c]),
             None if ground_truth is None else float(ground_truth[r, c]),
             r * size + c in highlights, f"r{r}c{c}"]
            for r in range(size) for c in range(size)]
    return dict(columns=OPENING_VALUE_COLUMNS, data=rows)


def board_figure(table, title, logits=False):
    import plotly.graph_objects as go
    size = table["data"][0][1]
    values = np.array([row[5] for row in table["data"]]).reshape(size, size)
    highlights = [row[2] for row in table["data"] if row[7]]
    label = "Logit" if logits else "Value"
    fig = go.Figure(go.Heatmap(
        z=values.tolist(), x=list(range(size)), y=list(range(size)),
        text=[[f"{x:+.2f}" for x in row] for row in values], texttemplate="%{text}",
        colorscale="Viridis" if logits else "RdBu", zmid=None if logits else 0.,
        colorbar=dict(title=label),
        hovertemplate=f"row=%{{y}}<br>col=%{{x}}<br>{label.lower()}=%{{z:.4f}}<extra></extra>"))
    if highlights:
        fig.add_trace(go.Scatter(
            x=[i % size for i in highlights], y=[i // size for i in highlights], mode="markers",
            marker=dict(symbol="square-open", size=34, color="red" if logits else "royalblue", line=dict(width=3)),
            name="Top-k" if logits else "Bottom-k", hoverinfo="skip"))
    fig.update_layout(title=title, xaxis=dict(title="Column", constrain="domain"),
                      yaxis=dict(title="Row", autorange="reversed", scaleanchor="x", scaleratio=1, constrain="domain"),
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def wandb_board_logs(tables):
    import wandb
    logs = {name: wandb.Table(**table) for name, table in tables.items()}
    for table_key, chart_key, title, logits in (
        ("model/values_after_black_moves_table", "model/values_after_black_moves_heatmap", "Value-Head After Each Possible Black Opening Move (Bottom-k in Blue)", False),
        ("model/logits_heatmap_top_k_table", "model/logits_heatmap_top_k", "Logits Heatmap (Top-k Highlighted)", True),
        ("muzero/model/latent_values_after_black_moves_table", "muzero/model/latent_values_after_black_moves_heatmap", "Learned-Latent Value After Each Black Opening (White perspective)", False),
    ):
        if table_key in tables:
            logs[chart_key] = wandb.Plotly(board_figure(tables[table_key], title, logits))
    return logs


# =============================================================================
# Training and replay metrics
# Source: src/nanoalphazero/research/muzero/metrics.py
# =============================================================================

"""AlphaZero metric names with explicit root/sequence and collection semantics."""


RENAMES = {
    "loss": "total_loss", "policy_loss": "loss_pi", "value_loss": "loss_v",
    "reward_loss": "muzero/loss_r", "grad_norm": "norms/grad_norm",
    "updates": "runner_state/n_updates", "seconds": "timing/cycle_seconds",
    "elapsed_seconds": "timing/training_seconds", "cycle_wall_seconds": "loop/loop_duration",
    "diagnostics/opening_value_mse": "hex_perf/mse_vs_perfect",
    "diagnostics/opening_value_sign_accuracy": "hex_perf/sign_accuracy",
    "staging/overwritten_fresh": "anomalies/selfplay_buffer/eviction_n_isvalid_fresh",
    "staging/exploration_excluded": "selfplay_buffer/n_excluded_exploration",
    "drain/inserted_positions": "muzero/drain/inserted_positions",
    "drain/remaining_fresh": "muzero/drain/remaining_fresh",
}
AZ_PREFIXES = (
    "norms/", "timing/", "loop/", "warmup/", "runner_state/", "selfplay/", "selfplay-reward/",
    "selfplay_buffer", "selfplay-intr/", "train_batch/", "training/", "drain/", "entropy/", "wdl/",
    "loss_per_step_", "errors/", "anomalies/", "openings_", "stats/", "debug/",
    "model/", "params-logits/", "hex_perf/", "hex_eval/",
)


def standardize(metrics):
    """One boundary for local JSON and W&B; extras cannot scatter namespaces."""
    result = {}
    for name, value in metrics.items():
        name = RENAMES.get(name, name)
        if name not in ("cycle", "total_loss", "loss_pi", "loss_v") and not name.startswith((*AZ_PREFIXES, "muzero/")):
            name = "muzero/" + name
        if name in result:
            raise ValueError(f"Metric collision: {name}")
        result[name] = value
    return result


def _mean(values, mask):
    return float(np.sum(np.where(mask, values, 0)) / max(np.sum(mask), 1))


def batch_metrics(batch, allows_draws):
    """AZ counters count B root positions, never B*(K+1) unroll targets."""
    values = np.asarray(batch["value"])[:, 0]
    valid = np.asarray(batch["policy_mask"])[:, 0]
    info = {k: np.asarray(v) for k, v in batch["sample_info"].items()}
    fresh = info["is_fresh_i8"]
    produced = (fresh == 1) & info["is_from_selfplay"]
    invalid = ~produced | info["is_exploration"] | (info["is_pending_reward_i8"] == 1)
    # A zero boundary bootstrap is legitimate, including in no-draw games.
    invalid_no_draw = invalid | ((values == 0) & info["terminal"])
    output = {
        "train_batch/n_reward1": int(np.sum(values == 1)),
        "train_batch/n_rewardneg1": int(np.sum(values == -1)),
        "train_batch/n_reward0": int(np.sum(values == 0)),
        "train_batch/n_valid": int(np.sum(valid)), "train_batch/n_invalid": int(np.sum(~valid)),
        "train_batch/n_from_selfplay_samples": int(np.sum(info["is_from_selfplay"])),
        "train_batch/n_is_fresh_i8_eq1": int(np.sum(fresh == 1)),
        "train_batch/n_is_fresh_i8_eq0": int(np.sum(fresh == 0)),
        "train_batch/n_unique_game_ids": int(len(np.unique(info["game_id"][valid]))),
        "anomalies/train_batch/n_invalid_term_at_step0": int(np.sum(valid & info["terminal"] & (info["ep_termination_step"] == 0))),
        "anomalies/train_batch/n_invalid_term_at_step1": int(np.sum(valid & info["terminal"] & (info["ep_termination_step"] == 1))),
        "anomalies/train_batch/n_is_fresh_i8_gte_2": int(np.sum(fresh >= 2)),
        "anomalies/train_batch/n_not_is_fresh_i8_but_valid": int(np.sum(valid & (fresh != 1))),
        "anomalies/train_batch/n_invalid_mismatches_chess": int(np.sum(~valid) - np.sum(invalid)),
        "anomalies/train_batch/n_invalid_mismatches_hex": int(np.sum(~valid) - np.sum(invalid_no_draw)),
        "debug/batch_checksum": float(np.sum(info["row_id"])),
        "muzero/train_batch/n_bootstrapped_roots": int(np.sum(valid & ~info["terminal"])),
        "muzero/train_batch/n_non_outcome_values": int(np.sum(valid & ~np.isin(values, [-1, 0, 1]))),
        "muzero/train_batch/n_real_targets": int(np.sum(batch["policy_mask"])),
        "muzero/train_batch/n_absorbing_targets": int(np.sum(np.asarray(batch["value_mask"]) & ~np.asarray(batch["policy_mask"]) & info["terminal"][:, None])),
        "muzero/train_batch/n_boundary_bootstraps": int(np.sum(np.asarray(batch["value_mask"]) & ~np.asarray(batch["policy_mask"]) & ~info["terminal"][:, None])),
    }
    suffix = "envallowsdraws" if allows_draws else "envforbidsdraws"
    output[f"train_batch/n_is_invalid_{suffix}_doublecheck"] = int(np.sum(invalid if allows_draws else invalid_no_draw))
    return output


def collection_metrics(episodes, global_step, allows_draws):
    """Aggregate actual transitions in the collection, excluding padded work.

    AZ logs its last asynchronous step; synchronous collection aggregates the
    whole collection. Names/units match, but the aggregation window is explicit.
    """
    e = {k: np.asarray(v) for k, v in episodes.items()}
    real, terminal, lengths = e["valid"], e["terminal"], e["length"]
    steps = np.broadcast_to(np.arange(real.shape[1]), real.shape)
    term_steps = lengths[terminal] - 1
    last = steps == lengths[:, None] - 1
    just_terminal = real & last & terminal[:, None]
    legal = e["next_legal_count"]
    rewards = e["reward"]
    output = {
        "selfplay/global_step": global_step,
        "selfplay/ep_term_step_min": int(np.min(term_steps)) if len(term_steps) else -1,
        "selfplay/ep_term_step_max": int(np.max(term_steps)) if len(term_steps) else -1,
        "selfplay/ep_term_step_avg": float(np.mean(term_steps)) if len(term_steps) else -1.,
        "selfplay/p1_wins": int(np.sum(terminal & (e["final_rewards"][:, 0] == 1))),
        "selfplay/p2_wins": int(np.sum(terminal & (e["final_rewards"][:, 1] == 1))),
        "selfplay/p_just_tied": int(np.sum(terminal & np.all(e["final_rewards"] == 0, -1))),
        "selfplay/n_legal_moves_min": int(np.min(legal[real])),
        "selfplay/n_legal_moves_max": int(np.max(legal[real])),
        "selfplay/n_legal_moves_avg": _mean(legal, real),
        "selfplay/n_legal_moves_avg_early": _mean(legal, real & (steps <= 10)),
        "selfplay/n_legal_moves_avg_mid": _mean(legal, real & (steps > 10) & (steps <= 30)),
        "selfplay/n_legal_moves_avg_late": _mean(legal, real & (steps > 30)),
        "selfplay/ep_step_min": int(np.min(lengths - 1)),
        "selfplay/ep_step_max": int(np.max(lengths - 1)),
        "selfplay/ep_step_std": float(np.std(lengths - 1)),
        "selfplay-reward/valid_1s_aft_term": int(np.sum(just_terminal & (np.abs(rewards) == 1))),
        "selfplay-reward/valid_0s_no_term": int(np.sum(real & ~just_terminal & (rewards == 0))),
        "selfplay_buffer/n_all_just_terminated": int(np.sum(lengths[terminal])),
        "selfplay_buffer/n_rows_just_terminated": int(np.sum(terminal)),
        "selfplay_buffer/n_hit_max_ep_step_ROWS": int(np.sum(lengths == real.shape[1])),
        "selfplay_buffer/n_hit_max_ep_step_ENTRIES": int(np.sum(lengths[lengths == real.shape[1]])),
        "selfplay_buffer-comp/n_es_term": int(np.sum(terminal)),
        "selfplay_buffer-comp/n_entries_to_update": int(np.sum(lengths[terminal])),
    }
    if allows_draws:
        output["selfplay-reward/valid_0s_aft_term"] = int(np.sum(just_terminal & (rewards == 0)))
    else:
        output["errors/selfplay-reward/invalid_0s_aft_term"] = int(np.sum(just_terminal & (rewards == 0)))
        output["errors/selfplay-reward/invalid_1s_no_term"] = int(np.sum(real & ~just_terminal & (rewards != 0)))
    logits = e["raw_policy_logits"]
    logits = logits - np.max(logits, -1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), -1, keepdims=True))
    policy = e["policy"]
    kl = np.sum(policy * (np.log(np.maximum(policy, 1e-30)) - log_probs), -1)
    selected = kl[real]
    output.update({"selfplay-intr/n_kl_close_to_zero": int(np.sum(selected < 1e-5)),
                   "selfplay-intr/n_kl_le_one": int(np.sum(selected <= 1)),
                   "selfplay-intr/kl_mean": float(np.mean(selected)),
                   "selfplay-intr/kl_std": float(np.std(selected)),
                   "selfplay-intr/kl_max": float(np.max(selected))})
    for name, q in (("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("p95", 95)):
        output[f"selfplay-intr/kl_{name}"] = float(np.percentile(selected, q))
    peak = np.zeros_like(real)
    peak[np.arange(len(real)), np.argmax(np.where(real, kl, -np.inf), -1)] = True
    peak &= terminal[:, None]
    others = real & terminal[:, None] & ~peak
    diff = np.abs(e["predicted_value"] - e["value"])
    for name, values in (("value_diff", diff), ("kl_divergence", kl)):
        output[f"selfplay_buffer/{name}_peak_interest"] = _mean(values, peak)
        output[f"selfplay_buffer/{name}_others"] = _mean(values, others)
    output["selfplay_buffer/num_peak_samples"] = int(np.sum(peak))
    if policy.shape[-1] <= 100:
        for i in range(policy.shape[-1]):
            output[f"openings_explore/{i}"] = int(np.sum((e["action"][:, 0] == i) & e["exploration"][:, 0]))
            output[f"openings_exploit/{i}"] = int(np.sum((e["action"][:, 0] == i) & ~e["exploration"][:, 0]))
    return output


def histogram_data(episodes, batch, consumed=None):
    e, info = episodes, batch["sample_info"]
    valid = np.asarray(batch["policy_mask"])[:, 0]
    arrays = {
        "selfplay/opening_actions_distribution": np.asarray(e["action"])[:, 0],
        "selfplay/opening_actions_distribution_not_explore": np.asarray(e["action"])[:, 0][~np.asarray(e["exploration"])[:, 0]],
        "train_batch/ep_step_distribution": np.asarray(info["ep_step"])[valid],
        "train_batch/opening_actions_distribution": np.asarray(info["action"])[valid & (np.asarray(info["ep_step"]) == 0)],
        "train_batch/sampled_row_ids_distribution": np.asarray(info["row_id"])[valid],
        "train_batch/sampled_game_ids_distribution": np.asarray(info["game_id"])[valid],
    }
    if "col_id" in info:
        arrays["train_batch/sampled_col_ids_distribution"] = np.asarray(info["col_id"])[valid]
    logits = np.asarray(e["raw_policy_logits"])
    logits = logits - np.max(logits, -1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), -1, keepdims=True))
    policies = np.asarray(e["policy"])
    kl = np.sum(policies * (np.log(np.maximum(policies, 1e-30)) - log_probs), -1)
    arrays["selfplay-intr/current_kl_divergence_distribution"] = kl[np.asarray(e["valid"])]
    if consumed is not None:
        for label, field in (("col_ids", "col_id"), ("valid_col_ids", "col_id"),
                             ("row_ids", "row_id"), ("valid_row_ids", "row_id"),
                             ("ep_step_all", "ep_step"), ("valid_game_ids", "game_id")):
            arrays[f"selfplay_buffer-comp-histogram/{label}_distribution"] = np.asarray(consumed[field])
    return {name: values for name, values in arrays.items() if len(values)}


# =============================================================================
# Model diagnostics using real observations
# Source: src/nanoalphazero/research/muzero/inspection.py
# =============================================================================

"""Compose production value diagnostics with MuZero initial inference."""


def opening_head_tables(model, params, env, config, output):
    """Log small numeric tables for dynamic charts, never rendered images."""
    import json

    size = config["boardsize"]
    count = size * size
    output.mkdir(parents=True, exist_ok=False)
    root = env.init_dummy_estate(count)
    latent, logits, initial_value = model.apply({"params": params}, env.observe(root, root.current_player), method=model.initial)
    following = env.step(root, jnp.arange(count))
    _, after_logits, after_value = model.apply(
        {"params": params}, env.observe(following, following.current_player), method=model.initial)
    _, reward, latent_logits, latent_value = model.apply(
        {"params": params}, latent, jnp.arange(count), method=model.recurrent)
    arrays = dict(empty_policy_logits=np.asarray(logits[0]), after_policy_logits=np.asarray(after_logits),
                  after_value=np.asarray(after_value), latent_policy_logits=np.asarray(latent_logits),
                  latent_value=np.asarray(latent_value), predicted_reward=np.asarray(reward),
                  after_legal_mask=np.asarray(following.legal_action_mask))
    np.savez_compressed(output / "heads.npz", **arrays)
    truth = perfect_play_values(size).reshape(-1)
    reply_rows = [[opening, action // size, action % size,
                   float(arrays["after_policy_logits"][opening, action]),
                   float(arrays["latent_policy_logits"][opening, action]),
                   bool(arrays["after_legal_mask"][opening, action])]
                  for opening in range(count) for action in range(count)]
    tables = {
        "model/values_after_black_moves_table": board_table(
            arrays["after_value"].reshape(size, size), truth.reshape(size, size), np.argsort(arrays["after_value"])[:size]),
        "model/logits_heatmap_top_k_table": board_table(
            arrays["empty_policy_logits"].reshape(size, size), highlights=np.argsort(-arrays["empty_policy_logits"])[:size]),
        "muzero/model/latent_values_after_black_moves_table": board_table(
            arrays["latent_value"].reshape(size, size), truth.reshape(size, size), np.argsort(arrays["latent_value"])[:size]),
        "muzero/model/reply_policy_logits_table": dict(
            columns=["opening", "row", "col", "policy_logit", "latent_policy_logit", "legal"], data=reply_rows),
    }
    (output / "tables.json").write_text(json.dumps(tables, allow_nan=False) + "\n")
    import jax
    probabilities = jax.nn.softmax(logits[0])
    entropy = float(-jnp.sum(probabilities * jnp.log(probabilities + 1e-8)))
    value_probs = jax.nn.softmax(after_value)
    metrics = {"model/init_board_value": float(initial_value[0]),
               "model/value_head_empty_board_entropy": float(-jnp.sum(value_probs * jnp.log(value_probs + 1e-8))),
               "model/policy_head_empty_board_entropy": entropy,
               "params-logits/init_logits_norm": float(jnp.linalg.norm(logits[0])),
               "params-logits/init_logits_std": float(jnp.std(logits[0])),
               "params-logits/init_logits_max": float(jnp.max(logits[0])),
               "params-logits/init_logits_min": float(jnp.min(logits[0])),
               "params-logits/init_logits_entropy": entropy,
               "params-logits/init_logits_entropy_its": entropy / np.log(2),
               "params-logits/num_moves_over_5pct": int(jnp.sum(probabilities > .05))}
    return tables, metrics


def inspect_position_values(model, params, env, config):

    def apply(variables, observation, legal, deterministic=True):
        _, logits, value = model.apply(variables, observation, method=model.initial)
        return logits, value

    state = SimpleNamespace(params=params, apply_fn=apply)
    env_id = config["env_id"]
    if env_id.startswith("hex"):
        _run_hex_diagnostics(state, env, config)
        size = config["boardsize"]
        root = env.init_dummy_estate(size * size)
        following = env.step(root, jnp.arange(size * size))
        _, _, value = model.apply({"params": params}, env.observe(following, following.current_player), method=model.initial)
        truth = perfect_play_values(size).reshape(-1)
        return {"hex_perf/mse_vs_perfect": float(jnp.mean((value - truth) ** 2)),
                "hex_perf/sign_accuracy": float(np.mean(np.sign(np.asarray(value)) == np.sign(truth)))}
    if env_id == "tic_tac_toe":
        _run_ttt_diagnostics(state, env, config)
    elif env_id == "connect_four":
        _run_connect4_diagnostics(state, env, config)
    elif env_id.startswith("go_"):
        _run_go_diagnostics(state, env, config)
    return {}


# =============================================================================
# Optional solver evaluation of actual decisions
# Source: src/nanoalphazero/research/muzero/decisions.py
# =============================================================================

"""Evaluation-only exact Hex 4 decisions on varied held-out midgame positions."""


def summarize(records):
    winning = [r for r in records if r["root_value"] == 1]
    result = {"positions": len(records), "winning_positions": len(winning)}
    for mode in ("policy", "search"):
        result[f"{mode}_winning_preserved"] = sum(r[f"{mode}_action_value"] == 1 for r in winning)
        result[f"{mode}_winning_accuracy"] = result[f"{mode}_winning_preserved"] / max(len(winning), 1)
        result[f"{mode}_action_value_mean"] = float(np.mean([r[f"{mode}_action_value"] for r in records]))
        result[f"{mode}_q_mse"] = float(np.mean([(r[f"{mode}_q"] - r[f"{mode}_action_value"]) ** 2 for r in records]))
        result[f"{mode}_reward_mse"] = float(np.mean([(r[f"{mode}_predicted_reward"] - r[f"{mode}_true_reward"]) ** 2 for r in records]))
    result["search_fixes_policy"] = sum(r["search_action_value"] > r["policy_action_value"] for r in winning)
    result["search_breaks_policy"] = sum(r["search_action_value"] < r["policy_action_value"] for r in winning)
    result["predicted_q_gain"] = float(np.mean([r["search_q"] - r["policy_q"] for r in records]))
    result["true_action_value_gain"] = float(np.mean([r["search_action_value"] - r["policy_action_value"] for r in records]))
    result["root_value_mse"] = float(np.mean([(r["predicted_root_value"] - r["root_value"]) ** 2 for r in records]))
    return result


def evaluate_decisions(model, params, env, config, pool, output, positions=256, seed=840017):
    """No solver results leave this evaluation routine for training or search.

    Uses explicit DFPN proofs, not MoHex's move-selection strength. Restrict
    this baseline to Hex 4; larger-board exact-solve cost is not assumed safe.
    """
    if config["env"] != "hex4":
        raise ValueError("Exact midgame decision evaluation is currently Hex 4 only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    batch = len(pool.engines)
    rng = np.random.default_rng(seed)
    records, seen = [], set()
    search = make_search(model, env, config)
    policy = make_search(model, env, config, policy_only=True)
    initial = jax.jit(lambda obs: model.apply({"params": params}, obs, method=model.initial))
    recurrent = jax.jit(lambda latent, action: model.apply({"params": params}, latent, action, method=model.recurrent))

    def solve(engine, history):
        engine.clear()
        for ply, action in enumerate(history):
            engine.play("b" if ply % 2 == 0 else "w", int(action))
        response = engine.query(f"dfpn-solve-state {'b' if len(history) % 2 == 0 else 'w'}").strip().lower()
        if response not in ("black", "white"):
            raise RuntimeError(f"DFPN did not prove a winner: {response!r}")
        return 0 if response == "black" else 1

    with ThreadPoolExecutor(max_workers=batch) as workers:
        for group in range(10000):
            if len(records) >= positions:
                break
            depth = 2 + group % 9
            histories = np.stack([rng.permutation(16)[:depth] for _ in range(batch)])
            state = env.init_dummy_estate(batch)
            for ply in range(depth):
                state = env.step(state, jnp.asarray(histories[:, ply]))
            observation = env.observe(state, state.current_player)
            observed = np.asarray(observation)
            active = []
            for i in range(batch):
                identity = observed[i].tobytes()
                if bool(state.terminated[i]) or identity in seen:
                    continue
                if len(records) + len(active) >= positions:
                    break
                seen.add(identity)
                active.append(i)
            if not active:
                continue
            actor = depth % 2
            root_futures = {i: workers.submit(solve, pool.engines[i], histories[i]) for i in active}
            roots = {i: future.result() for i, future in root_futures.items()}
            latent, _, values = initial(observation)
            actions = {"policy": policy(jax.random.PRNGKey(0), state, params, 0., batch).action,
                       "search": search(jax.random.PRNGKey(0), state, params, 0., batch).action}
            rows = {i: {"history": histories[i].tolist(), "depth": depth, "actor": actor,
                        "root_value": 1 if roots[i] == actor else -1,
                        "predicted_root_value": float(values[i])} for i in active}
            for mode, action in actions.items():
                if not bool(jnp.all(state.legal_action_mask[jnp.arange(batch), action])):
                    raise RuntimeError("Illegal real decision-evaluation action")
                _, reward, _, next_value = recurrent(latent, action)
                following = env.step(state, action)
                pending = {i: workers.submit(solve, pool.engines[i], [*histories[i], int(action[i])])
                           for i in active if not bool(following.terminated[i])}
                for i in active:
                    actual_reward = float(following.rewards[i, actor])
                    if bool(following.terminated[i]):
                        action_value = int(np.sign(actual_reward))
                    else:
                        action_value = 1 if pending[i].result() == actor else -1
                    if roots[i] != actor and action_value == 1:
                        raise RuntimeError("Inconsistent DFPN proofs: a lost position has a winning action")
                    rows[i].update({f"{mode}_action": int(action[i]), f"{mode}_action_value": action_value,
                                    f"{mode}_q": float(reward[i] - config["discount"] * next_value[i]),
                                    f"{mode}_predicted_reward": float(reward[i]),
                                    f"{mode}_true_reward": actual_reward})
            with (output / "positions.jsonl").open("a") as stream:
                for row in rows.values():
                    records.append(row)
                    stream.write(json.dumps(row) + "\n")
        else:
            raise RuntimeError("Failed to generate enough unique nonterminal positions")
    result = summarize(records)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"seed": seed, "positions": positions,
        "depths": list(range(2, 11)), "solver_command": "dfpn-solve-state",
        "source": "uniform random legal prefixes, deduplicated by observation",
        "scope": "Hex 4 only; exact action values on these sampled states"}, indent=2) + "\n")
    return result


# =============================================================================
# Held-out predictions and policy/search playing strength
# Source: src/nanoalphazero/research/muzero/evaluation.py
# =============================================================================

"""Real-environment matches and held-out unroll diagnostics."""


def make_evaluator(model, env, config, reference=None):
    search = make_search(model, env, config)
    policy = make_search(model, env, config, policy_only=True)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    @functools.partial(jax.jit, static_argnums=(3, 4),
                       in_shardings=(replicated, data, replicated), out_shardings=data)
    def match(params, openings, key, mode0, mode1):
        n = openings.shape[0]
        state = env.init_dummy_estate(n)
        for d in range(openings.shape[1]):
            state = env.step(state, openings[:, d])

        def step(carry, _):
            state, rewards, finished, key = carry
            key, ak = jax.random.split(key)
            random = jnp.argmax(jnp.where(legal_actions(state), jax.random.gumbel(ak, (n, env.num_actions)), -jnp.inf), -1)
            actions = {"random": random}
            for mode in set((mode0, mode1)) - {"random"}:
                if mode == "alphazero":
                    fn, reference_params = reference
                    actions[mode] = fn(ak, state, reference_params, 0., n).action
                else:
                    fn = search if mode == "search" else policy
                    actions[mode] = fn(ak, state, params, 0., n).action
            action = jnp.where(state.current_player == 0, actions[mode0], actions[mode1]).astype(jnp.int32)
            illegal = ~finished & ~legal_actions(state)[jnp.arange(n), action]
            next_state = env.step(state, action)
            newly = ~finished & next_state.terminated
            rewards = jnp.where(newly[:, None], next_state.rewards, rewards)
            return (next_state, rewards, finished | next_state.terminated, key), (action, illegal)

        (_, rewards, finished, _), (actions, illegal) = jax.lax.scan(
            step, (state, state.rewards, state.terminated, key), None, length=config["max_steps"]
        )
        return {"rewards": rewards, "finished": finished, "actions": actions.T, "illegal": illegal.T}

    def evaluate(params, key, directory, opening_plies=2):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        openings = all_opening_actions(env, config, plies=opening_plies)
        # Hex/TTT opening counts are multiples of four at depth two. General
        # environments may need padding; discard duplicated rows when scoring.
        count = len(openings)
        pad = (-count) % config["devices"]
        padded = np.concatenate([openings, np.resize(openings, (pad, opening_plies))])
        np.save(directory / "openings.npy", openings)
        scores = {}
        pairings = [("policy", "random"), ("random", "policy"),
                             ("search", "random"), ("random", "search"),
                             ("search", "policy"), ("policy", "search")]
        if reference is not None:
            pairings += [("policy", "alphazero"), ("alphazero", "policy"),
                         ("search", "alphazero"), ("alphazero", "search")]
        for mode0, mode1 in pairings:
            key, mk = jax.random.split(key)
            started = time.monotonic()
            out = jax.device_get(match(params, jnp.asarray(padded), mk, mode0, mode1))
            if np.any(out["illegal"]):
                raise RuntimeError("Illegal evaluation action")
            label = f"{mode0}-vs-{mode1}"
            np.savez_compressed(directory / f"{label}.npz", **out)
            r = out["rewards"][:count, 0]
            scores[label] = dict(wins=int(np.sum(r > 0)), draws=int(np.sum(r == 0)),
                                 losses=int(np.sum(r < 0)), unfinished=int(np.sum(~out["finished"][:count])),
                                 seconds=time.monotonic() - started)
        (directory / "summary.json").write_text(json.dumps(scores, indent=2) + "\n")
        return scores
    return evaluate


@functools.lru_cache(maxsize=16)
def _heldout_function(model):
    # Keep one callable alive across monitoring events. Recreating jitted
    # partials here otherwise recompiles each unroll shape at every evaluation.
    @jax.jit
    def evaluate(params, batch):
        _, metrics = loss(model, params, batch)
        return metrics, unroll_predictions(model, params, batch)
    return evaluate


def heldout_metrics(model, params, episodes, key, unrolls=(1, 3, 5, 10)):
    result = {}
    for unroll in unrolls:
        batch = sequences(episodes, key, unroll)
        metrics, (_, values, rewards) = _heldout_function(model)(params, batch)
        result.update({f"heldout/unroll{unroll}/{k}": float(v) for k, v in metrics.items()})
        # Report real states separately: absorbing zeros otherwise make longer
        # unroll averages look deceptively easier.
        def masked_mean(x, mask):
            return float(jnp.sum(jnp.where(mask, x, 0.)) / jnp.maximum(jnp.sum(mask), 1))
        result[f"heldout/unroll{unroll}/real_value_mse"] = masked_mean((values - batch["value"]) ** 2, batch["policy_mask"])
        # Production-style replay excludes exploratory starts. Keep the mixed
        # held-out metric, but expose this distribution difference explicitly.
        # These groups are defined by the sampled root, not by each later ply.
        exploration = batch["sample_info"]["is_exploration"]
        for label, roots in (("exploration", exploration), ("nonexploration", ~exploration)):
            mask = batch["policy_mask"] & roots[:, None]
            prefix = f"heldout/unroll{unroll}/{label}_root"
            result[f"{prefix}_count"] = int(jnp.sum(roots))
            result[f"{prefix}_real_value_mse"] = masked_mean((values - batch["value"]) ** 2, mask)
            result[f"{prefix}_k0_value_mse"] = masked_mean(
                (values[:, 0] - batch["value"][:, 0]) ** 2, roots)
        positive = batch["reward_mask"] & (batch["reward"] > 0)
        result[f"heldout/unroll{unroll}/nonzero_reward_count"] = int(jnp.sum(positive))
        result[f"heldout/unroll{unroll}/positive_reward_prediction"] = masked_mean(rewards, positive)
        result[f"heldout/unroll{unroll}/real_reward_mse"] = masked_mean((rewards - batch["reward"]) ** 2, batch["policy_mask"][:, :-1])
        for k in range(unroll + 1):
            result[f"heldout/unroll{unroll}/k{k}_real_value_mse"] = masked_mean((values[:, k] - batch["value"][:, k]) ** 2, batch["policy_mask"][:, k])
    return result


def evaluation_main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a MuZero checkpoint in real games")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphazero", type=Path)
    parser.add_argument("--mohex-engine-path")
    parser.add_argument("--mohex-engine-config", default="default")
    parser.add_argument("--opening-plies", type=int, default=2)
    parser.add_argument("--platform", choices=("cpu", "tpu"), default="tpu")
    args = parser.parse_args()
    pool = None
    if args.mohex_engine_path:
        config = metadata(args.checkpoint)
        if not config["env"].startswith("hex"):
            raise ValueError("MoHex requires a Hex checkpoint")
        size = int(config["env"][3:])
        pool = MoHexPool(args.mohex_engine_path, args.mohex_engine_config, size, size * size)
    try:
        _run_evaluation(args, pool)
    finally:
        if pool is not None:
            pool.close()


def _run_evaluation(args, pool):
    # JAX has been imported but device initialization has not occurred here.
    jax.config.update("jax_platforms", args.platform)
    params, config = load_params(args.checkpoint)
    if len(jax.devices()) != 4:
        raise ValueError("Evaluation requires four devices")
    env = make_env(config)
    if list(env.obs_shape) != config["obs_shape"] or env.num_actions != config["num_actions"]:
        raise ValueError("Environment/checkpoint interface mismatch")
    model = build_model(config)
    reference = None
    if args.alphazero:
        ref_params, metadata = load_checkpoint(str(args.alphazero))
        ref_config = apply_checkpoint_model_config(CONFIG_FACTORIES[config["env"]](), metadata)
        if ref_config["env_id"] != config["env_id"]:
            raise ValueError("AlphaZero checkpoint environment mismatch")
        ref_config.update(game_obs_shape=env.obs_shape, game_num_actions=env.num_actions, enable_sharding=False)
        ref_model, _ = make_model(ref_config, jax.random.PRNGKey(0))
        reference = make_mcts(ref_config, env, ref_model), ref_params
    scores = make_evaluator(model, env, config, reference)(
        params, jax.random.PRNGKey(config["seed"] + 2000000), args.output, args.opening_plies
    )
    import hashlib
    manifest = {"checkpoint": str(args.checkpoint), "config": config,
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "alphazero": str(args.alphazero) if args.alphazero else None,
                "matched": "same MuZero parameters for policy/search; AlphaZero is an unmatched strength reference",
                "platform": args.platform, "opening_plies": args.opening_plies}
    if args.alphazero:
        manifest["alphazero_sha256"] = hashlib.sha256(args.alphazero.read_bytes()).hexdigest()
    if pool is not None:
        inspection_config = {**CONFIG_FACTORIES[config["env"]](), "enable_sharding": True}
        manifest["mohex"] = {"executable": str(pool.executable), "config": pool.config.read_text(),
                              "sha256": hashlib.sha256(pool.executable.read_bytes()).hexdigest()}
        for mode in ("search", "policy"):
            evaluator = HexTrainingEvaluator(1, pool, args.output / f"mohex-{mode}",
                                              board_size=inspection_config["boardsize"])
            fn = make_search(model, env, config, policy_only=mode == "policy")
            scores[f"mohex-{mode}"] = evaluator.run_if_due(1, fn, env, inspection_config, params)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


# =============================================================================
# Optional stopping gate for repeated MoHex opening coverage
# Source: src/nanoalphazero/research/muzero/convergence.py
# =============================================================================

"""Host-only stopping gate for repeated, actual MoHex opening evaluations."""


class OpeningCoverageGate:
    def __init__(self, required=0):
        self.required = required
        self.streak = 0

    def observe(self, metrics):
        if not self.required:
            return False
        prefixes = ("hex_eval/", "hex_eval_policy/")
        if not all(prefix + "perfect_opening_total" in metrics for prefix in prefixes):
            return False
        success = all(
            metrics[prefix + "perfect_opening_total"] > 0
            and metrics.get(prefix + "perfect_opening_wins", -1)
            == metrics[prefix + "perfect_opening_total"]
            and metrics.get(prefix + "unscored", -1) == 0
            for prefix in prefixes
        )
        self.streak = self.streak + 1 if success else 0
        return self.streak >= self.required


# =============================================================================
# Four-device training loop and command line
# Source: src/nanoalphazero/research/muzero/cli.py
# =============================================================================

"""Installed MuZero research training and smoke entry point."""


def resolve(raw):
    name = raw.get("env", "hex4")
    base = CONFIG_FACTORIES[name]()
    config = dict(env=name, env_id=base["env_id"], width=128, depth=2, unroll=5,
                  discount=1.0, max_steps=base["game_max_steps"],
                  roots=base["mcts_num_root_considered"], survivors=base["mcts_num_survivors"],
                  exploration_moves=base["num_exploratory_moves"],
                  selfplay_batch_size=32, train_batch_size=32, heldout_batch_size=0, replay_batches=8,
                  updates_per_cycle=4, cycles=2, seed=0, learning_rate=1e-3,
                  weight_decay=1e-4, devices=4, platform="tpu", wandb=False,
                  save_checkpoints=False, checkpoint_period=0, eval_period=10, defaults="alphazero",
                  warmup_updates=0, decay_kernels_only=False,
                  exploration_mode="fixed", root_temperature=1.0, remat_unroll=False, remat_blocks=False,
                  value_scale=1.0, maxvisit_init=50., rescale_values=False,
                  hex_eval_period=0, opening_coverage_streak=0, diagnostic_period=base["diagnostic_period"],
                  network="vector", activation=base.get("katago_activation", "mish"),
                  use_rvgl=base.get("katago_use_rvgl", True), decision_eval_positions=0,
                  data_pipeline="episodes", staging_batches=8,
                  consume_size=base["selfplay_buffer_consume_size"],
                  replay_positions=base["replay_buffer_total_size"],
                  replay_warmup_cycles=(base["replay_buffer_warmup_steps"] + base["game_max_steps"] - 1)
                  // base["game_max_steps"])
    if raw.get("defaults", "alphazero") == "alphazero":
        train_batch = 4096 if name == "hex5" else base["train_batch_size"]
        config.update(
            width=base["conv_width"], depth=base["conv_depth"],
            selfplay_batch_size=base["selfplay_batch_size"], train_batch_size=train_batch,
            learning_rate=base["learning_rate"], weight_decay=base["weight_decay"],
            updates_per_cycle=base["cycle_n_train"],
            # Production num_iters counts self-play steps, not updates.
            cycles=base["num_iters"] // base["cycle_n_selfplay"],
            replay_batches=max(1, base["replay_buffer_total_size"] //
                               (base["selfplay_batch_size"] * base["game_max_steps"])),
            warmup_updates=base["lr_warmup_steps"],
            decay_kernels_only=base.get("weight_decay_kernels_only", False),
            exploration_mode="random_switch",
            root_temperature=base.get("exp_root_temperature", 1.) if base.get("exp_use_root_temperature") else 1.,
            value_scale=base["mcts_value_scale"], maxvisit_init=base["mcts_maxvisit_init"],
            rescale_values=base["mcts_rescale_values"],
            eval_period=base["eval_period"],
        )
    elif raw.get("defaults") != "pilot_v1":
        raise ValueError("defaults must be alphazero or pilot_v1")
    unknown = set(raw) - set(config)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config.update(raw)
    # Diagnostics have independent RNGs and never enter training replay.
    # Zero preserves the historical full self-play batch diagnostic size.
    if config["heldout_batch_size"] == 0:
        config["heldout_batch_size"] = config["selfplay_batch_size"]
    if config["network"] == "vector":
        # These architectural options belong to the spatial production modules.
        # Record the vector tower's actual choices rather than unused defaults.
        config.update(activation="relu", use_rvgl=False)
    for key in ("width", "unroll", "max_steps", "roots", "survivors", "selfplay_batch_size",
                "train_batch_size", "heldout_batch_size", "replay_batches", "updates_per_cycle", "cycles", "devices"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["survivors"] > config["roots"]:
        raise ValueError("survivors exceeds roots")
    if config["root_temperature"] <= 0 or config["warmup_updates"] < 0:
        raise ValueError("Invalid root temperature or warmup")
    if not isinstance(config["save_checkpoints"], bool) or config["checkpoint_period"] < 0:
        raise ValueError("save_checkpoints must be boolean and checkpoint_period nonnegative")
    if not isinstance(config["remat_unroll"], bool):
        raise ValueError("remat_unroll must be boolean")
    if not isinstance(config["remat_blocks"], bool):
        raise ValueError("remat_blocks must be boolean")
    if (type(config["opening_coverage_streak"]) is not int
            or config["opening_coverage_streak"] < 0):
        raise ValueError("opening_coverage_streak must be a nonnegative integer")
    if config["opening_coverage_streak"] and (
        not config["env"].startswith("hex") or config["hex_eval_period"] <= 0
    ):
        raise ValueError("Opening coverage stopping requires periodic Hex evaluation")
    if config["exploration_mode"] not in ("fixed", "random_switch"):
        raise ValueError("Unknown exploration mode")
    if config["network"] not in ("vector", "spatial"):
        raise ValueError("Unknown network")
    if config["data_pipeline"] not in ("episodes", "staged"):
        raise ValueError("Unknown data pipeline")
    if min(config["staging_batches"], config["consume_size"], config["replay_positions"]) < 1:
        raise ValueError("Staging/replay sizes must be positive")
    if config["data_pipeline"] == "staged" and config["consume_size"] > config["selfplay_batch_size"] * config["staging_batches"] * config["max_steps"]:
        raise ValueError("consume_size exceeds staging capacity")
    if config["replay_warmup_cycles"] < 0 or config["replay_positions"] < config["consume_size"]:
        raise ValueError("Invalid replay warmup or capacity")
    if not 0 < config["discount"] <= 1:
        raise ValueError("discount must be in (0, 1]")
    if name == "hex5" and config["train_batch_size"] != 4096:
        raise ValueError("Hex 5 requires train_batch_size = 4096")
    if config["platform"] == "tpu" and config["devices"] != 4:
        raise ValueError("TPU experiments require exactly four devices")
    if config["cycles"] > 10 and not config["wandb"]:
        raise ValueError("Long runs require W&B")
    return config


def training_main():
    args = standalone_train_parser().parse_args()
    raw = standalone_train_settings(args)
    config = resolve(raw)
    if args.print_config:
        print(json.dumps(config, indent=2))
        return
    if args.output is None:
        args.output = standalone_output(config)
    print(f"Run output: {args.output}", flush=True)
    engine_pool = None
    if config["hex_eval_period"] or config["decision_eval_positions"]:
        if not config["env"].startswith("hex") or not args.hex_eval_engine_path:
            raise ValueError("Periodic MoHex evaluation requires Hex and --hex-eval-engine-path")
        # Match production: external processes must start before libtpu/JAX.
        size = int(config["env"][3:])
        engine_pool = MoHexPool(args.hex_eval_engine_path, args.hex_eval_engine_config, size, size * size)
    try:
        _run(args, config, engine_pool)
    finally:
        if engine_pool is not None:
            engine_pool.close()


def _run(args, config, engine_pool):
    # Selection precedes imports that initialize production's device mesh.
    import os
    os.environ["JAX_PLATFORMS"] = config["platform"]
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.training.train_state import TrainState

    devices = jax.devices()
    if len(devices) != config["devices"]:
        raise ValueError(f"Expected {config['devices']} devices, found {devices}")
    for key in ("selfplay_batch_size", "train_batch_size", "heldout_batch_size", "consume_size"):
        if config[key] % len(devices):
            raise ValueError(f"{key} must be divisible by device count")
    args.output.mkdir(parents=True, exist_ok=False)
    mesh = jax.sharding.Mesh(np.array(devices), ("data",))
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    env = make_env(config)
    config.update(obs_shape=list(env.obs_shape), num_actions=env.num_actions,
                  architecture=f"{config['network']}-v1", target="signed-return-boundary-bootstrap",
                  latent_normalization="minmax", latent_normalization_epsilon=1e-5,
                  dynamics_gradient_scale=0.5, policy_loss="kl_target_prediction",
                  value_loss="scalar_mse", reward_loss="scalar_mse", max_grad_norm=1.0,
                  optimizer="adamw", initial_warmup_lr=1e-6,
                  metric_schema="alphazero-v1",
                  replay_sampling=("uniform_consumed_position" if config["data_pipeline"] == "staged"
                                   else "uniform_episode_then_uniform_real_position"))
    inspection_config = {**CONFIG_FACTORIES[config["env"]](), "enable_sharding": True}
    if config["roots"] > env.num_actions:
        raise ValueError("roots exceeds action count")
    if args.resume and metadata(args.resume) != config:
        raise ValueError("Resume requires identical resolved configuration")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"command": __import__("sys").argv, "devices": [str(d) for d in devices],
                "jax": jax.__version__, "started": datetime.now(timezone.utc).isoformat(),
                **standalone_provenance()}
    if engine_pool is not None:
        manifest["mohex"] = {"executable": str(engine_pool.executable),
                              "config": engine_pool.config.read_text(),
                              "sha256": hashlib.sha256(engine_pool.executable.read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    standalone_snapshot(args.output, manifest)
    model = build_model(config)
    key, init_key = jax.random.split(jax.random.PRNGKey(config["seed"]))
    params = model.init(init_key, jnp.zeros((1, *env.obs_shape)), jnp.zeros(1, jnp.int32))["params"]
    rate = config["learning_rate"]
    if config["warmup_updates"]:
        rate = optax.join_schedules([
            optax.linear_schedule(1e-6, rate, config["warmup_updates"]),
            optax.constant_schedule(rate)], [config["warmup_updates"]])
    decay_mask = None
    if config["decay_kernels_only"]:
        from flax.traverse_util import flatten_dict, unflatten_dict
        decay_mask = unflatten_dict({p: p[-1] in ("kernel", "kernel_3x3", "kernel_1x1")
                                     for p in flatten_dict(params)})
    state = TrainState.create(apply_fn=model.apply, params=params,
                              tx=optax.chain(optax.clip_by_global_norm(1.0),
                                             optax.adamw(rate, weight_decay=config["weight_decay"], mask=decay_mask)))
    state = jax.device_put(state, replicated)
    search = make_search(model, env, config)
    collect = jax.jit(make_collect(env, model, search, config), out_shardings=data)
    diagnostic_collect = collect
    if config["heldout_batch_size"] != config["selfplay_batch_size"]:
        diagnostic_collect = jax.jit(make_collect(
            env, model, search, {**config, "selfplay_batch_size": config["heldout_batch_size"]}),
            out_shardings=data)
    dummy = jax.eval_shape(collect, state.params, key)
    example = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), dummy)
    staging_state = None
    if config["data_pipeline"] == "staged":
        init_staging, stage_add, consume, drain, replay = make_staging(config)
        staging_shape = jax.eval_shape(init_staging, example)
        staging_sharding = jax.tree.map(lambda x: data if x.ndim else replicated, staging_shape)
        staging_state = jax.jit(init_staging, out_shardings=staging_sharding)(example)
        batch_shape = jax.eval_shape(consume, staging_state, key)[1]
        example = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), batch_shape)
        stage_add = jax.jit(stage_add, out_shardings=(staging_sharding, replicated), donate_argnums=(0,))
    else:
        replay = make_replay(config)
    shape = jax.eval_shape(replay.init, example)
    replay_sharding = jax.tree.map(lambda x: data if x.ndim else replicated, shape)
    replay_state = jax.jit(replay.init, out_shardings=replay_sharding)(example)
    add = jax.jit(replay.add, donate_argnums=(0,), out_shardings=replay_sharding)
    if staging_state is not None:
        drain = jax.jit(drain, out_shardings=(staging_sharding, replay_sharding, replicated, replicated),
                        donate_argnums=(0, 1))
    sample = jax.jit(lambda s, k: jax.tree.map(lambda x: x[:, 0], replay.sample(s, k).experience), out_shardings=data)
    make_batch = jax.jit(functools.partial(sequences, unroll=config["unroll"]), out_shardings=data)
    train = jax.jit(functools.partial(train_step, model, remat=config["remat_unroll"]), in_shardings=(replicated, data),
                    out_shardings=(replicated, replicated), donate_argnums=(0,))
    start_cycle = 0
    if args.resume:
        template = {"train": state, "replay": vars(replay_state), "key": key, "cycle": jnp.array(0)}
        if staging_state is not None:
            template["staging"] = staging_state
        loaded, _ = load(args.resume, template)
        state = jax.device_put(loaded["train"], replicated)
        replay_state = jax.device_put(replay_state.replace(**loaded["replay"]), replay_sharding)
        key, start_cycle = loaded["key"], int(loaded["cycle"])
        if staging_state is not None:
            staging_state = jax.device_put(loaded["staging"], staging_sharding)
    run = None
    if config["wandb"]:
        import wandb
        run = wandb.init(project="nanoAlphaZero-muzero", config=config,
                         name=f"{config['env']}-{config['network']}-two-rung-seed{config['seed']}-{args.output.name}")
        manifest["wandb_url"] = run.url
        run.define_metric("runner_state/n_updates")
        run.define_metric("runner_state/n_selfplay_steps")
        run.summary["stats/num_params"] = sum(x.size for x in jax.tree.leaves(state.params))
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    evaluator = make_evaluator(model, env, config)
    hex_evaluators = {}
    if engine_pool is not None:
        for mode in ("search", "policy"):
            hex_evaluators[mode] = HexTrainingEvaluator(
                config["hex_eval_period"], engine_pool, args.output / f"mohex-{mode}",
                board_size=inspection_config["boardsize"])
    policy_search = make_search(model, env, config, policy_only=True)
    # Separate seeds and episodes; these records are never inserted into replay.
    heldout = None
    if config["eval_period"] and config["cycles"] >= config["eval_period"]:
        if args.resume:
            heldout_path = args.resume.parent / "heldout-initial-selfplay.npz"
            if not heldout_path.exists():
                raise ValueError("Resume needs the original held-out episodes beside its checkpoint")
            with np.load(heldout_path) as saved:
                heldout = jax.device_put({name: saved[name] for name in saved.files}, data)
        else:
            heldout = diagnostic_collect(state.params, jax.random.PRNGKey(config["seed"] + 1000000))
        jax.block_until_ready(heldout)
        np.savez_compressed(args.output / "heldout-initial-selfplay.npz", **jax.device_get(heldout))
        initial_metrics = heldout_metrics(model, state.params, heldout, jax.random.PRNGKey(12345))
        (args.output / "heldout-initial-metrics.json").write_text(json.dumps(initial_metrics, indent=2) + "\n")
        baseline = evaluator(state.params, jax.random.PRNGKey(config["seed"] + 2000000),
                             args.output / f"eval-{start_cycle:06d}")
        print(json.dumps({"initial_evaluation": baseline}), flush=True)

    def collect_and_insert(params, rng, staged, replayed):
        collection_started = time.monotonic()
        rng, ck = jax.random.split(rng)
        episodes = collect(params, ck)
        jax.block_until_ready(episodes)
        collection_seconds = time.monotonic() - collection_started
        insertion_started = time.monotonic()
        if bool(jnp.any(episodes["illegal"])):
            raise RuntimeError("Illegal real self-play action")
        data_metrics = {}
        if staged is None:
            episodes = {**episodes, "game_id": replayed.current_index.astype(jnp.uint32) * config["selfplay_batch_size"]
                        + jnp.arange(config["selfplay_batch_size"], dtype=jnp.uint32) + 1}
            replayed = add(replayed, jax.tree.map(lambda x: x[:, None], episodes))
        else:
            staged, data_metrics = stage_add(staged, episodes)
            if int(data_metrics["staging/overwritten_fresh"]) or int(data_metrics["staging/malformed_episodes"]):
                raise RuntimeError("Staging overflow or malformed episode; refusing to train")
            staged, replayed, rng, drain_metrics = drain(staged, replayed, rng)
            data_metrics = {**data_metrics, **drain_metrics}
        data_metrics = {
            k: float(v) if jnp.issubdtype(v.dtype, jnp.floating) else int(v) for k, v in data_metrics.items()}
        # Synchronize insertion for truthful stage timings, including the legacy
        # episode pipeline where there are no drain counters to synchronize it.
        jax.block_until_ready(replayed.current_index)
        data_metrics.update({
            "muzero/timing/collection_seconds": collection_seconds,
            "muzero/timing/replay_insert_seconds": time.monotonic() - insertion_started,
        })
        return episodes, rng, staged, replayed, data_metrics

    warmup_start = time.monotonic()
    if staging_state is not None and not args.resume:
        for warmup in range(config["replay_warmup_cycles"]):
            warmup_cycle_start = time.monotonic()
            episodes, key, staging_state, replay_state, data_metrics = collect_and_insert(
                state.params, key, staging_state, replay_state)
            record = standardize({"warmup/loop_n": warmup + 1,
                                  "warmup/loop_duration": time.monotonic() - warmup_cycle_start,
                                  "real_transitions": int(jnp.sum(episodes["length"])),
                                  "runner_state/n_updates": int(state.step), **data_metrics})
            with (args.output / "warmup.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if warmup == 0 or (warmup + 1) % 10 == 0:
                print(json.dumps(record), flush=True)
            if run:
                run.log(record, step=warmup)
        manifest["replay_warmup_cycles"] = config["replay_warmup_cycles"]
    if run:
        run.summary["stats/warmup_duration"] = time.monotonic() - warmup_start
    learning_started = time.monotonic()
    stop_requested = False
    coverage_gate = OpeningCoverageGate(config["opening_coverage_streak"])
    coverage_reached = False
    cycle = start_cycle
    for cycle in range(start_cycle + 1, config["cycles"] + 1):
        cycle_start = time.monotonic()
        episodes, key, staging_state, replay_state, data_metrics = collect_and_insert(
            state.params, key, staging_state, replay_state)
        if not bool(replay.can_sample(replay_state)):
            raise RuntimeError("Replay has no complete consumable batch; increase collection/warmup")
        optimizer_started = time.monotonic()
        for _ in range(config["updates_per_cycle"]):
            key, rk, bk = jax.random.split(key, 3)
            sampled = sample(replay_state, rk)
            batch = sampled if staging_state is not None else make_batch(sampled, bk)
            state, metrics = train(state, batch)
        if cycle == start_cycle + 1:
            manifest["sharding"] = {
                "selfplay_observation": str(episodes["observation"].sharding),
                "replay_observation": str(replay_state.experience["observation"].sharding),
                "training_observation": str(batch["observation"].sharding),
                "training_local_shapes": [list(s.data.shape) for s in batch["observation"].addressable_shards],
            }
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["muzero/timing/optimizer_seconds"] = time.monotonic() - optimizer_started
        stop_requested = args.stop_file is not None and args.stop_file.exists()
        final_cycle = cycle == config["cycles"] or stop_requested
        metrics.update(data_metrics)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise RuntimeError(f"Nonfinite training metrics: {metrics}")
        metrics.update(cycle=cycle, updates=int(state.step),
                       seconds=time.monotonic() - cycle_start,
                       elapsed_seconds=time.monotonic() - started,
                       completed_fraction=float(jnp.mean(episodes["terminal"])),
                       real_transitions=int(jnp.sum(episodes["length"])))
        metrics.update({
            "norms/current_lr": float(rate(state.step - 1)) if callable(rate) else float(rate),
            "muzero/selfplay/episode_length_mean": float(jnp.mean(episodes["length"])),
            "muzero/selfplay/episode_length_std": float(jnp.std(episodes["length"].astype(jnp.float32))),
            "muzero/selfplay/episode_length_min": int(jnp.min(episodes["length"])),
            "muzero/selfplay/episode_length_max": int(jnp.max(episodes["length"])),
            "muzero/selfplay/truncated_fraction": float(jnp.mean(~episodes["terminal"])),
            "runner_state/train_step": int(state.step),
            "runner_state/n_selfplay_steps": cycle * config["max_steps"],
            "norms/n_updates": int(state.step), "loop/loop_n": cycle,
            "debug/sample_rng_hash": float(jnp.sum(rk.astype(jnp.float32))),
            "debug/rbuf_current_index": int(replay_state.current_index),
        })
        metrics.update(batch_metrics(batch, inspection_config["env_allows_draws"]))
        priming_cycles = config["replay_warmup_cycles"] if staging_state is not None else 0
        metrics.update(collection_metrics(episodes, (cycle + priming_cycles) * config["max_steps"], inspection_config["env_allows_draws"]))
        if staging_state is not None:
            occupied = config["replay_positions"] // config["consume_size"] if bool(replay_state.is_full) else int(replay_state.current_index)
            metrics.update({"training/rbuf-n_is_valid": occupied * config["consume_size"],
                            "training/rbuf-n_is_fresh": occupied * config["consume_size"],
                            "training/spbuf_num_consumables": int(jnp.sum(staging_state.fresh)),
                            "training/n_slices_drained": data_metrics["drain/n_slices"]})
        diagnostic_tables = {}
        if cycle == 1 or final_cycle or (config["diagnostic_period"] and cycle % config["diagnostic_period"] == 0):
            metrics.update(inspect_position_values(model, state.params, env, inspection_config))
            if config["env"].startswith("hex"):
                diagnostic_tables, diagnostic_scalars = opening_head_tables(
                    model, state.params, env, inspection_config, args.output / f"diagnostics-{cycle:06d}")
                metrics.update(diagnostic_scalars)
        for mode, hex_evaluator in hex_evaluators.items():
            # This research-owned evaluator instance can force a final actual-
            # game evaluation without changing the production evaluator API.
            period = hex_evaluator.period
            if final_cycle and period:
                hex_evaluator.period = 1
            try:
                results = hex_evaluator.run_if_due(
                    cycle, search if mode == "search" else policy_search, env,
                    inspection_config, state.params, train_step=int(state.step))
            finally:
                hex_evaluator.period = period
            prefix = "hex_eval/" if mode == "search" else "hex_eval_policy/"
            metrics.update({k.replace("hex_eval/", prefix): v for k, v in results.items()})
        coverage_reached = coverage_gate.observe(metrics)
        metrics["muzero/opening_coverage_streak"] = coverage_gate.streak
        metrics["muzero/opening_coverage_reached"] = int(coverage_reached)
        final_cycle = final_cycle or coverage_reached
        if heldout is not None and (cycle % config["eval_period"] == 0 or final_cycle):
            metrics.update(heldout_metrics(model, state.params, heldout, jax.random.PRNGKey(12345)))
            fresh = diagnostic_collect(state.params, jax.random.PRNGKey(config["seed"] + 3000000 + cycle))
            fresh_metrics = heldout_metrics(model, state.params, fresh, jax.random.PRNGKey(12345))
            metrics.update({k.replace("heldout/", "heldout_current/"): v for k, v in fresh_metrics.items()})
            metrics["heldout_current/real_transitions"] = int(jnp.sum(fresh["length"]))
            np.savez_compressed(args.output / f"heldout-cycle-{cycle:06d}.npz", **jax.device_get(fresh))
            scores = evaluator(state.params, jax.random.PRNGKey(config["seed"] + 2000000),
                               args.output / f"eval-{cycle:06d}")
            for label, result in scores.items():
                total = result["wins"] + result["draws"] + result["losses"]
                metrics[f"eval/{label}/seat0_score"] = (result["wins"] + .5 * result["draws"]) / total
        if final_cycle and config["decision_eval_positions"]:
            result = evaluate_decisions(model, state.params, env, config, engine_pool,
                                        args.output / "exact-decisions", config["decision_eval_positions"])
            metrics.update({f"decisions/{k}": v for k, v in result.items()})
        metrics["elapsed_seconds"] = time.monotonic() - learning_started
        metrics["muzero/timing/run_seconds"] = time.monotonic() - started
        metrics["cycle_wall_seconds"] = time.monotonic() - cycle_start
        metrics["muzero/operator_stop_requested"] = int(stop_requested)
        metrics["loop/loop_total_duration"] = metrics["elapsed_seconds"]
        metrics = standardize(metrics)
        print(json.dumps(metrics), flush=True)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if run:
            consumed_info = None
            if staging_state is not None and data_metrics["drain/n_slices"]:
                last_slot = (replay_state.current_index - 1) % replay_state.experience["policy_mask"].shape[1]
                consumed_info = jax.tree.map(lambda x: x[:, last_slot], replay_state.experience["sample_info"])
            histograms = {name: wandb.Histogram(values) for name, values in histogram_data(episodes, batch, consumed_info).items()}
            run.log({**metrics, **histograms, **wandb_board_logs(diagnostic_tables)}, step=cycle + priming_cycles)
        if config["save_checkpoints"] and (
            final_cycle or
            (config["checkpoint_period"] > 0 and cycle % config["checkpoint_period"] == 0)
        ):
            saved = {"train": state, "replay": vars(replay_state), "key": key, "cycle": jnp.array(cycle)}
            if staging_state is not None:
                saved["staging"] = staging_state
            save(args.output / f"cycle-{cycle:06d}.safetensors",
                            saved, config)
        if stop_requested or coverage_reached:
            break
    manifest.update(finished=datetime.now(timezone.utc).isoformat(),
                    completed_cycles=cycle, completed_updates=int(state.step),
                    stop_reason=("operator_stop_file" if stop_requested else
                                 "opening_coverage_reached" if coverage_reached else "configured_cycles_complete"),
                    opening_coverage_streak=coverage_gate.streak,
                    parameter_count=sum(x.size for x in jax.tree.leaves(state.params)),
                    memory_stats=[d.memory_stats() for d in devices])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if run:
        run.summary["stats/total_learn_duration"] = time.monotonic() - learning_started
        run.finish()


if __name__ == "__main__":
    standalone_main()
