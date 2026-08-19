# nanoAlphaZero

nanoAlphaZero is a game-agnostic, high-performance implementation of AlphaZero. Despite its size, it can reach perfect play in games like Hex and grandmaster-level strength in chess.

demo: [play against the nets](https://nanoalphazero.wtedw.com/) 

<div align="center">
<img width="900" height="565" alt="elo_multi_hours_light-6-30-26" src="https://github.com/user-attachments/assets/f5e09015-dc98-453b-b953-748c10517038#gh-light-mode-only" />
<img width="900" height="565" alt="elo_multi_hours-6-30-26" src="https://github.com/user-attachments/assets/8f9bcf59-2b40-49b1-9b68-fad2fba1559b#gh-dark-mode-only" />
</div>

## How is this different?

- **It scales to chess.** Not a toy AlphaZero implementation. It can train a
  grandmaster level chess model in under 24h on a TPU.
- **Genuinely game-agnostic**. We validate the core logic across Hex, Connect4,
  Go, and chess, and demonstrate how to train AlphaZero on custom games of your
  own using a Colab notebook.
- **Training is one JAX function.** Self-play, MCTS, and training
  are fused into a single jitted call (`run_fn`).
- **It's dead simple to run.** Install `uv`, clone the repo,
  `uv run train --env chess`.
- **It's fast.** We use custom, TPU-native JAX environments ([pgx1](https://github.com/wtedw/pgx1)) that
  are orders of magnitude faster to run compared to the reference implementation.
  For MCTS search, we parallelize the sequential halving algorithm from Gumbel
  MuZero via [mctx](https://github.com/deepmind/mctx):

  | env      | pgx     | pgx1     | speedup | env/s (batch 4096) |
  | -------- | ------: | -------: | ------: | ------------------: |
  | go_9x9   | 80.5 ms | 0.535 ms | 150x    | 7.7M                |
  | go_19x19 | 656 ms  | 1.595 ms | 411x    | 2.6M                |
  | chess    | 904 ms  | 0.832 ms | 1087x   | 4.9M                |

> note: all code is optimized for TPUs; correctness on GPUs isn't guaranteed.

## Setup

On a fresh TPU VM:

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Clone the repo
git clone https://github.com/wtedw/nanoAlphaZero.git
cd nanoAlphaZero

# 3. Train (first run resolves + installs deps)
uv run train --env ttt
```

## Train

Train a model, then drop into an interactive game against it. The trained params
and resolved model configuration auto-save to
`artifacts/alphazero_<env>.safetensors`. Safetensors is the only supported
checkpoint format.

```bash
uv run train --env ttt
uv run train --env connect4
uv run train --env hex5
uv run train --env chess
```

Supported games:

| game        | envs              | status                                                     |
| ----------- | ----------------- | ---------------------------------------------------------- |
| Tic-Tac-Toe | `ttt`             | solid, reaches perfect play                                |
| Hex         | `hex4`–`hex9`     | solid up to 8x8 (`hex9` is less tested)                    |
| Chess       | `chess`           | solid, reaches strong play given enough compute            |
| Connect4    | `connect4`        | reaches perfect play outcomes, can struggle to maintain it |
| Go          | `go3`–`go9`       | trains well up to 9x9                                      |

Options:

| flag           | effect                                            |
| -------------- | ------------------------------------------------- |
| `--save PATH`  | custom save path for the checkpoint               |
| `--no-save`    | train only, don't write a checkpoint              |
| `--no-play`    | train only, skip the interactive game afterward   |
| `--enable-wandb` | log metrics and upload versioned model artifacts |

<details>
<summary>Watch it train (all in the terminal)</summary>

Metrics are logged straight to the terminal. It periodically prints an ASCII
loss curve and other env-specific diagnostics:

```text
➜  ~ uv run train --env connect4
Warmup finished in 26.1s.
Model has 9,458,308 parameters.
Starting training for 2000 cycles...
Cycle 1/2000 | 1.74s
  phase1 selfplay | p1_wins=317 p2_wins=222 ties=0 n_legal_avg_mid=6.77
  phase2 drain    | consumable=71585 slices=8
  phase3 train    | loss=0.8216 loss_v=0.5008 loss_pi=0.3209 batch[r+=4359 r-=3826 r0=7 valid=8192 invalid=0]
Cycle 2/2000 | 1.74s
  phase1 selfplay | p1_wins=297 p2_wins=204 ties=0 n_legal_avg_mid=6.79
  phase2 drain    | consumable=70214 slices=8
  phase3 train    | loss=0.8335 loss_v=0.4998 loss_pi=0.3337 batch[r+=4407 r-=3784 r0=1 valid=8192 invalid=0]
Cycle 3/2000 | 1.74s
  phase1 selfplay | p1_wins=299 p2_wins=229 ties=0 n_legal_avg_mid=6.78
  phase2 drain    | consumable=69670 slices=8
  phase3 train    | loss=0.8200 loss_v=0.4980 loss_pi=0.3220 batch[r+=4383 r-=3804 r0=5 valid=8192 invalid=0]

...

Cycle 48/2000 | 1.75s
  phase1 selfplay | p1_wins=167 p2_wins=129 ties=25 n_legal_avg_mid=6.11
  phase2 drain    | consumable=110212 slices=13
  phase3 train    | loss=0.9877 loss_v=0.3518 loss_pi=0.6359 batch[r+=3923 r-=3513 r0=756 valid=8192 invalid=0]
Cycle 49/2000 | 1.75s
  phase1 selfplay | p1_wins=162 p2_wins=120 ties=17 n_legal_avg_mid=6.09
  phase2 drain    | consumable=108795 slices=13
  phase3 train    | loss=0.9803 loss_v=0.3480 loss_pi=0.6323 batch[r+=3792 r-=3632 r0=768 valid=8192 invalid=0]
Cycle 50/2000 | 1.75s
  phase1 selfplay | p1_wins=182 p2_wins=122 ties=20 n_legal_avg_mid=5.94
  phase2 drain    | consumable=104636 slices=12
  phase3 train    | loss=0.9567 loss_v=0.3485 loss_pi=0.6083 batch[r+=3882 r-=3546 r0=764 valid=8192 invalid=0]

── loss over last 50 cycles ──
  1.3409 ┤                          ╭╮
  1.2984 ┤                        ╭─╯╰──╮
  1.2559 ┤                      ╭─╯     ╰╮
  1.2134 ┤                   ╭──╯        ╰──╮
  1.1709 ┤                  ╭╯              ╰╮
  1.1284 ┤                ╭─╯                ╰╮
  1.0859 ┤               ╭╯                   ╰───╮
  1.0434 ┤              ╭╯                        ╰──╮
  1.0009 ┤            ╭─╯                            ╰──╮
  0.9584 ┤          ╭─╯                                 ╰──
  0.9159 ┤       ╭──╯               ╭╮
  0.8734 ┤     ╭─╯                ╭─╯╰──╮
  0.8309 ┤╭╮ ╭─╯                ╭─╯     ╰─╮
  0.7884 ┤╯╰─╯               ╭──╯         ╰─╮
  0.7459 ┤                  ╭╯              ╰─╮
  0.7034 ┤                ╭─╯                 ╰─╮╭╮ ╭╮
  0.6609 ┤               ╭╯                     ╰╯╰─╯│
  0.6184 ┤              ╭╯                           ╰────╮
  0.5759 ┤             ╭╯                                 ╰
  0.5334 ┤            ╭╯
  0.4909 ┤───────╮  ╭─╯
  0.4484 ┤       ╰╭─╯──╮
  0.4059 ┤     ╭──╯    ╰──────────────────────╮
  0.3634 ┤    ╭╯                              ╰─────────╮
  0.3209 ┤────╯                                         ╰──
  ● total  ● value  ● policy

...

Cycle 1998/2000 | 1.75s
  phase1 selfplay | p1_wins=172 p2_wins=116 ties=40 n_legal_avg_mid=6.04
  phase2 drain    | consumable=105882 slices=12
  phase3 train    | loss=0.0731 loss_v=0.0186 loss_pi=0.0545 batch[r+=3409 r-=3007 r0=1776 valid=8192 invalid=0]
Cycle 1999/2000 | 1.75s
  phase1 selfplay | p1_wins=174 p2_wins=116 ties=21 n_legal_avg_mid=6.05
  phase2 drain    | consumable=109770 slices=13
  phase3 train    | loss=0.0647 loss_v=0.0153 loss_pi=0.0494 batch[r+=3400 r-=3085 r0=1707 valid=8192 invalid=0]
Cycle 2000/2000 | 1.75s
  phase1 selfplay | p1_wins=147 p2_wins=127 ties=33 n_legal_avg_mid=6.02
  phase2 drain    | consumable=106517 slices=13
  phase3 train    | loss=0.0692 loss_v=0.0164 loss_pi=0.0528 batch[r+=3268 r-=3120 r0=1804 valid=8192 invalid=0]

  ── ladder Elo over cycles 50..2000 ──
  2137.5776┤      ╭───────╮╭──╮╭───╮╭╮╭─────────────
  1959.4461┤    ╭─╯       ╰╯  ╰╯   ╰╯╰╯
  1781.3146┤  ╭─╯
  1603.1832┤╭─╯
  1425.0517┤│
  1246.9202┤│
  1068.7888┤│
  890.6573 ┤│
  712.5259 ┤│
  534.3944 ┤│
  356.2629 ┤│
  178.1315 ┤│
    0.0000 ┤╯
    ● total

  Training finished in 3627.1s.
  ✅ Saved model params to artifacts/alphazero_connect4.safetensors

  ==================================================
  Playing connect_four.  You are 'X' (player 1).
  Enter a column number (1-7) to drop your piece.
  Commands: undo, restart, quit
  ==================================================

     .  .  .  .  .  .  .
     .  .  .  .  .  .  .
     .  .  .  .  .  .  .
     .  .  .  .  .  .  .
     .  .  .  .  .  .  .
     .  .  .  .  .  .  .
     1  2  3  4  5  6  7

  Your move (X):
```

</details>

## Train a custom game in Colab

We can train AlphaZero in Colab on a completely new game by hoisting the core logic from this package
Run the notebook in Colab on a TPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wtedw/nanoAlphaZero/blob/main/alphazero_colab.ipynb)

> note: This notebook is a WIP, sparse on details, but works.

## Eval

### Chess

Run a tournament between nanoAlphaZero checkpoints, Searchless Chess, and
Stockfish:

```bash
uv sync --group dev

# fetch openings, reference models, and BayesElo
uv run assets fetch \
  eco-openings searchless-270m stockfish-16 bayeselo \
  desert-snowball-34400 desert-snowball-68800

# reproduce the 400-simulation tournament between a 24h trained model, 48 trained model and searchless270m
uv run eval \
  tournament-desert-snowball-checkpoints-vs-searchless270m-400sims-512games-per-pair

# score an existing games.pgn with BayesElo, e.g. reproducing the Searchless
# Chess paper's all-models-vs-Stockfish matchup
uv run bayeselo --pgn evals/tournament-searchless-all-vs-stockfish16-oracle-50ms-128games-per-pair/runs/20260818-072706-5dc47c444b/games.pgn
```


## Results

### Chess

Using a TPU v4-32 pod, we can train grandmaster-level chess models in under
24h with ~12 MCTS simulations.

But is it really grandmaster level?

Since there are hardly any reference players to run against on TPUs, we base
the "grandmaster-level" claim on beating Searchless Chess's 270M model, which
itself reached a 2895 Lichess Blitz Elo against human players.

Their models are ideal to play against because:
1. they're JAX/Haiku based
2. we can run batched tournaments against them on TPUs
3. they don't require search, so one model inference should reproduce the same
   strength

Here's how our
[10x256nbt models](https://github.com/wtedw/nanoAlphaZero/releases/tag/desert-snowball-1028-checkpoints-v1)
do against Searchless Chess 270M.

```text
Relative Elo vs Searchless Chess 270M (270M = 0)

Searchless 270M        +0 |
model34400 · 400 sims  +27 | #####
model34400 · 800 sims  +90 | ##################
model68800 · 400 sims  +99 | ####################
model68800 · 800 sims +177 | ###################################

Each # represents approximately 5 Elo.
```

| Checkpoint | Training | Search | W-D-L vs 270M | Score | Relative Elo |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model34400` | 24h | 400 sims | 205-144-163 | 54.1% | +27 |
| `model34400` | 24h | 800 sims | 256-131-125 | 62.8% | +90 |
| `model68800` | 48h | 400 sims | 261-131-120 | 63.8% | +99 |
| `model68800` | 48h | 800 sims | 325-103-84 | 73.5% | +177 |

Trained on a TPU v4-32 pod. Elo is BayesElo relative to the 270M opponent, not
an absolute human rating.

See the
[`400-simulation results`](evals/tournament-desert-snowball-checkpoints-vs-searchless270m-400sims-512games-per-pair/README.md)
and [tournament documentation](docs/chess-tournaments.md) for raw results,
configuration, resume, adjudication, scoring, and scheduler details.

### Hex

On a solved game like hex we can watch the value head acquire perfect play
directly. The trainer periodically prints its verdict on every Black opening
move next to the known perfect-play outcome. As training proceeds, `MSE vs
perfect` falls toward 0 and `sign accuracy` climbs to 1.000.

For example, these are the value head outputs for each opening once training finishes.
```
hex6
Perfect play (B=Black wins / .=Black loses), [x]=model sign mismatch:
   .  .  .  .  .  B
   .  B  B  B  B  B
   B  B  B  B  B  B
   B  B  B  B  B  B
   B  B  B  B  B  .
   B  .  .  .  .  .
  MSE vs perfect = 0.0004 | sign accuracy = 1.000


hex7
Perfect play (B=Black wins / .=Black loses), [x]=model sign mismatch:
   .  .  .  .  .  .  B
   .  .  B  .  B  B  B
   .  B  B  B  B  B  .
   B  B  B  B  B  B  B
   .  B  B  B  B  B  .
   B  B  B  .  B  .  .
   B  .  .  .  .  .  .
  MSE vs perfect = 0.0082 | sign accuracy = 1.000
```

Compare this to the known perfect play outcomes in Hex
- hex6 bottom row, right
- hex7 bottom row, middle
<img width="836" height="529" alt="Pasted image 20250507172600" src="https://github.com/user-attachments/assets/e64de723-1902-4135-982b-d3deb68073cd" />


`MSE vs perfect` is a good proxy for progress, but a near-zero MSE does not guarantee the model can reliably beat a strong opponent like MoHex, so we set configs to keep training well past that point.

<details>
<summary>hex7: value head progression</summary>

**Cycle 1** — value head is flat ~0, it knows nothing yet. `MSE 0.9996 | sign acc 0.673`

```text
Cycle 1/5000 | 8.40s

--- Hex value-head after each Black opening (value = White-to-move perspective; negative => Black-winning) ---
  -0.00 -0.00 -0.00 -0.00 -0.00 -0.00 -0.00
  +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 -0.00
  +0.00 -0.00 -0.00 -0.00 -0.00 -0.00 -0.00
  +0.00 -0.00 -0.00 -0.00 -0.00 -0.00 -0.00
  +0.00 -0.00 -0.00 -0.00 -0.00 -0.00 -0.00
  -0.00 -0.00 -0.00 -0.00 -0.00 -0.00 -0.00
  +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00
  Perfect play (B=Black wins / .=Black loses), [x]=model sign mismatch:
  [.][.][.][.][.][.] B
   .  . [B] . [B][B] B
   .  B  B  B  B  B [.]
  [B] B  B  B  B  B  B
   .  B  B  B  B  B [.]
   B  B  B [.] B [.][.]
  [B] .  .  .  .  .  .
  MSE vs perfect = 0.9996 | sign accuracy = 0.673
```

**Cycle 100** — coarse map forming, many sign errors remain. `MSE 0.6118 | sign acc 0.837`

```text
Cycle 100/5000 | 8.39s

--- Hex value-head after each Black opening (value = White-to-move perspective; negative => Black-winning) ---
  +0.46 +0.43 +0.42 +0.59 +0.61 +0.65 -0.59
  +0.05 +0.21 +0.41 +0.49 +0.15 -0.36 -0.18
  +0.16 +0.19 +0.20 +0.45 -0.78 +0.06 +0.47
  -0.14 -0.29 -0.59 -0.73 -0.74 -0.39 -0.08
  +0.59 -0.04 -0.70 +0.62 -0.28 -0.22 +0.24
  -0.31 -0.67 -0.02 +0.04 +0.04 +0.18 +0.22
  -0.57 +0.53 +0.46 +0.54 +0.44 +0.35 +0.39
  Perfect play (B=Black wins / .=Black loses), [x]=model sign mismatch:
   .  .  .  .  .  .  B
   .  . [B] . [B] B  B
   . [B][B][B] B [B] .
   B  B  B  B  B  B  B
   .  B  B [B] B  B  .
   B  B  B  . [B] .  .
   B  .  .  .  .  .  .
  MSE vs perfect = 0.6118 | sign accuracy = 0.837
```

**Cycle 250** — magnitudes sharpening toward ±1, only 2 sign errors left. `MSE 0.1272 | sign acc 0.959`

```text
Cycle 250/5000 | 8.40s

--- Hex value-head after each Black opening (value = White-to-move perspective; negative => Black-winning) ---
  +0.92 +0.88 +0.80 +0.88 +0.87 +0.82 -0.94
  +0.28 -0.11 -0.59 -0.26 -0.96 -1.00 -0.99
  +0.24 -0.98 -0.97 -0.99 -1.00 -0.98 +0.54
  -0.59 -0.86 -1.00 -1.00 -1.00 -0.95 -0.89
  +0.16 -0.98 -1.00 -0.99 -0.99 -0.99 +0.40
  -0.95 -1.00 -0.94 +0.63 -0.78 +0.89 +0.53
  -0.94 +0.74 +0.94 +0.91 +0.91 +0.91 +0.93
  Perfect play (B=Black wins / .=Black loses), [x]=model sign mismatch:
   .  .  .  .  .  .  B
   . [.] B [.] B  B  B
   .  B  B  B  B  B  .
   B  B  B  B  B  B  B
   .  B  B  B  B  B  .
   B  B  B  .  B  .  .
   B  .  .  .  .  .  .
  MSE vs perfect = 0.1272 | sign accuracy = 0.959
```

**Cycle 500** — all signs correct, magnitudes nearly saturated. `MSE 0.0260 | sign acc 1.000`

```text
Cycle 500/5000 | 8.40s

--- Hex value-head after each Black opening (value = White-to-move perspective; negative => Black-winning) ---
  +1.00 +0.99 +0.99 +0.99 +0.99 +0.98 -0.99
  +0.65 +0.08 -0.89 +0.98 -0.97 -1.00 -0.99
  +0.95 -1.00 -1.00 -0.99 -1.00 -0.99 +0.90
  -0.87 -0.94 -1.00 -1.00 -1.00 -0.94 -0.53
  +0.93 -0.99 -1.00 -1.00 -1.00 -1.00 +0.99
  -0.94 -1.00 -0.98 +0.97 -0.90 +0.98 +0.99
  -0.97 +0.98 +1.00 +1.00 +1.00 +1.00 +1.00
  Perfect play (B=Black wins / .=Black loses), [x]=model sign mismatch:
   .  .  .  .  .  .  B
   .  .  B  .  B  B  B
   .  B  B  B  B  B  .
   B  B  B  B  B  B  B
   .  B  B  B  B  B  .
   B  B  B  .  B  .  .
   B  .  .  .  .  .  .
  MSE vs perfect = 0.0260 | sign accuracy = 1.000
```

**Cycle 1000** — pretty much close to perfect play outcomes. `MSE 0.0007 | sign acc 1.000`

```text
Cycle 1000/5000 | 8.40s

--- Hex value-head after each Black opening (value = White-to-move perspective; negative => Black-winning) ---
  +1.00 +1.00 +1.00 +0.99 +1.00 +1.00 -0.99
  +0.99 +0.99 -0.90 +0.99 -1.00 -1.00 -0.99
  +0.98 -1.00 -1.00 -1.00 -1.00 -1.00 +0.91
  -0.95 -0.93 -1.00 -1.00 -1.00 -0.97 -0.96
  +0.98 -1.00 -1.00 -0.98 -1.00 -1.00 +1.00
  -0.99 -1.00 -0.99 +0.97 -0.96 +0.99 +1.00
  -0.99 +0.99 +0.99 +0.99 +1.00 +1.00 +1.00
  Perfect play (B=Black wins / .=Black loses), [x]=model sign mismatch:
   .  .  .  .  .  .  B
   .  .  B  .  B  B  B
   .  B  B  B  B  B  .
   B  B  B  B  B  B  B
   .  B  B  B  B  B  .
   B  B  B  .  B  .  .
   B  .  .  .  .  .  .
  MSE vs perfect = 0.0007 | sign accuracy = 1.000
```

</details>


## Todo
- Evaluate the full MCTX tournament backend across larger TPU batch sizes
- Verify larger hex boards still hit perfect play
- Test Go models against reference opponent

## Acknowledgements
This project would not have been possible without the amazing work of the following:
- **MCTX** — search algorithm ([paper](https://openreview.net/forum?id=bERaNdoegnO))
- **PGX** — game environments
- **Flashbax** — replay buffers
- **Scaling Scaling Laws** — experiments & model architecture
- **KataGo** — many methods
- **[TPU Starter](https://github.com/ayaka14732/tpu-starter)** — TPU setup guide
- Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)
