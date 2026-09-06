# Disk usage audit — 2026-09-05

**Subsequent action requested by the user:** removed all 49 MuZero checkpoint
files/archives, reclaiming 6,949,089,280 allocated bytes (6.47 GiB). Free disk
rose to about 16 GiB. Saving is now off by default, including the final save.
The inventory below describes usage before that deletion. Raw evaluation
datasets and training histories remain available.

Measured from the repository root with `du -x -B1 --max-depth=1 .`, a file
inventory deduplicated by inode, and safetensors headers. Sizes below use GiB
(2^30 bytes); live Hex6 output grows while training runs.

| Location | Allocated size | Main contents |
|---|---:|---|
| `artifacts/` | 6.87 GiB | 6.78 GiB MuZero outputs plus existing AlphaZero assets |
| `logs/` | 1.80 GiB | Earlier Hex-reasoner research; 467 checkpoints account for 1.67 GiB |
| `.venv/` | 1.45 GiB | Installed dependencies, including libtpu and jaxlib |
| `wandb/` | 0.40 GiB | Local run histories, debug logs, numeric tables and Plotly JSON |
| `evals/`, `src/`, `tests/` | about 15 MiB | Configs, selected results, implementation and tests |
| **Repository total** | **10.54 GiB** | About 11.31 billion allocated bytes |

Git metadata is shared with `/home/tedpsw/nanoAlphaZero/.git` because this is a
worktree. Git objects occupy approximately 18 MiB; Git history is not the cause
of the local disk pressure.

## MuZero contribution

| Category | Allocated size | Files |
|---|---:|---:|
| Uncompressed checkpoints | 4.75 GiB | 46 |
| Losslessly compressed checkpoint archives | 1.72 GiB | 3 |
| Held-out self-play datasets | 0.26 GiB | 89 |
| Raw evaluation arrays | 0.009 GiB | 1228 |
| Other MuZero files | 0.036 GiB | 919 at inventory time |

The main implementation choice responsible for this is saving the entire replay
and staging buffers in every resumable checkpoint. This is much larger than
model-only saving. For the original Hex5 seed-0 final checkpoint:

- Replay: **1,599,488,005 bytes** (96.1% of the checkpoint).
- Staging: 45,129,737 bytes.
- Model plus optimizer: 19,150,080 bytes; model weights alone are about 6.4 MB.

Both cycle 850 and cycle 1700 are retained, each 1.664 GB uncompressed. Together
they occupy 3.10 GiB. The newer Hex5 seed-1 archive occupies 0.827 GiB, the Hex4
seed-1 archive 0.494 GiB, and the full-size Hex6 smoke archive 0.400 GiB.
The Hex6 smoke remains large despite its small replay capacity because its
8192 × 8 × 36 staging payload occupies **1,096,351,753 bytes** uncompressed.

## Active Hex6 growth

The full Hex6 replay checkpoint is estimated at **3.53 GB uncompressed**, from
the smoke tensor shapes scaled to 2,048,000 replay positions. Its first held-out
batch occupies **62,730,288 bytes**. Saving similarly sized batches every 50
cycles for 3500 cycles could add approximately **4.4 GB** of periodic datasets;
actual size depends on compression as play changes. Model and optimizer alone
occupy about 145 MB. This archival policy is expensive even though routine
charts are numeric JSON and no raster images are uploaded.

The live run keeps its existing settings. Changing checkpoint or held-out
retention should be an explicit future storage design change, not silent loss
of experiment evidence. Suitable changes are small model-only snapshots for
routine evaluation, less frequent full resumable snapshots, and a separately
configured held-out batch size. Compact sequence storage could also avoid
materializing overlapping policy-target windows in replay.

## Storage actions already taken

Before this audit, the three newly completed checkpoints were losslessly gzip
archived. Their full decompressed SHA-256 hashes were verified and recorded;
the original checkpoint byte streams remain recoverable.

To provide room for Hex6, 2,474,268,817 bytes of reproducible pip HTTP download
cache were cleared from `/home/tedpsw/.cache/pip/http-v2`. Separately, 317 old,
closed TPU driver logs were gzip archived after checking that no process held
them open. Their 1,794,589,117 original bytes are preserved in 246,909,815 bytes
of archives, saving 1,547,679,302 bytes. Each decompressed stream was hash-checked.
The inventory and hashes are in
`artifacts/muzero/progression-20260905-tools/closed-tpu-log-archives.jsonl`.
No experiment checkpoint or raw evaluation dataset was deleted. No UV cache
override or cleanup was used. Older Hex-reasoner outputs remain untouched.

The root filesystem is 97 GiB total and had about 8.4 GiB free after these
actions. Most machine-wide usage lies outside this 10.54 GiB repository.
