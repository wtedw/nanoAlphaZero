# nanoAlphaZero

nanoAlphaZero is quite possibly the strongest single-file implementation of AlphaZero out there. Despite its small footprint, it achieves perfect-play results for any solvable game, and with enough train-time and test-time compute, it reaches super-grandmaster Elo in chess. 

<div align="center">
<img width="900" height="565" alt="elo_multi_hours_light-6-30-26" src="https://github.com/user-attachments/assets/f5e09015-dc98-453b-b953-748c10517038#gh-light-mode-only" />
<img width="900" height="565" alt="elo_multi_hours-6-30-26" src="https://github.com/user-attachments/assets/8f9bcf59-2b40-49b1-9b68-fad2fba1559b#gh-dark-mode-only" />
</div>

It's also game-agnostic: point it at any two-player board game, adjust the model size, and it learns to play.

> note: this is a WIP.
> - all code is optimized for TPUs, including our custom PGX fork ([wtedw/pgx](https://github.com/wtedw/pgx/tree/dcb18cb))
> - chess reaches strong play after about 200k updates; smaller games converge much sooner
> - custom games require their own PGX env implementation

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
uv run alphazero.py --env ttt
```

## Train

Train a model, then drop into an interactive game against it. The trained params
auto-save to `artifacts/alphazero_<env>.pkl`.

```bash
uv run alphazero.py --env ttt
uv run alphazero.py --env connect4
uv run alphazero.py --env hex5
uv run alphazero.py --env chess
```

Supported games:

| game        | envs              | status                                                     |
| ----------- | ----------------- | ---------------------------------------------------------- |
| Tic-Tac-Toe | `ttt`             | solid, reaches perfect play                                |
| Hex         | `hex4`–`hex9`     | solid up to 8x8 (`hex9` is less tested)                    |
| Chess       | `chess`           | solid, reaches strong play given enough compute            |
| Connect4    | `connect4`        | reaches perfect play outcomes, can struggle to maintain it |
| Go          | `go3`–`go9`       | recently added, not yet tested                             |

Options:

| flag           | effect                                            |
| -------------- | ------------------------------------------------- |
| `--save PATH`  | custom save path for the checkpoint               |
| `--no-save`    | train only, don't write a checkpoint              |
| `--no-play`    | train only, skip the interactive game afterward   |

### Watch it train (all in the terminal)

Metrics are logged straight to the terminal. It periodically prints an ASCII
loss curve and other env-specific diagnostics:

```text
➜  ~ uv run alphazero.py --env connect4
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
  ✅ Saved model params to artifacts/alphazero_connect4.pkl

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

## Play

Play against an already-trained model without retraining — loads the checkpoint
and plays in the terminal:

```bash
uv run alphazero.py --env connect4 --play-only
uv run alphazero.py --env hex5 --play-only --load artifacts/alphazero_hex5.pkl
```

Options:

| flag             | effect                                                       |
| ---------------- | ------------------------------------------------------------ |
| `--play-only`    | skip training; load a checkpoint and play                    |
| `--play-both`    | skip training; load a checkpoint and play as both sides      |
| `--load PATH`    | checkpoint to load (defaults to the save path)               |
| `--play-as 1\|2` | you move first (`1`, default) or the model moves first (`2`) |


In-game commands: enter a move (connect4: column `1-7`; ttt/hex: cell number or
`row col`), or type `undo`, `restart`, `quit`.

> Interactive play supports ttt / connect4 / hex. Chess is not supported yet [todo].

## Results

### Chess

Chess models can be trained with just ~12 MCTS simulations per move. For
reference, the original AlphaZero used ~800.

The graphs below show score vs Stockfish at 2800 Elo* as training scales.
Each point is measured over 492 games, played as both player 1 and player 2.

<img width="1185" height="765" alt="sf_score" src="https://github.com/user-attachments/assets/e86402a2-4054-47e1-a37d-ebc31110ccfd" />


All models were trained on a TPUv4-32:

| model          | train time | updates |
| -------------- | ---------- | ------- |
| `model_12800`  | 25h        | 256k    |
| `model_6400`   | 12h        | 128k    |
| `model_3200`   | 6h         | 64k     |

Stockfish settings:

```json
{"UCI_LimitStrength": "true", "UCI_Elo": 2800, "Threads": 1, "Hash": 16}
```

> *A fair comparison to Stockfish is difficult.
> For one, Stockfish is CPU based, and requires calibration to operate at the prescribed Elo.
> Two, our MCTS is jitted to a fixed simulation count per move, so the search budget is always constant.
> Treat the numbers as relative and the Stockfish opponent as a point of reference.

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
- Incorporate the full gumbel/muzero MCTS from MCTX
- Verify larger hex boards still hit perfect play
- Test Go models against reference opponent

## Acknowledgements
This project would not have been possible without the amazing work of the following:
- **MCTX** — search algorithm ([paper](https://openreview.net/forum?id=bERaNdoegnO))
- **PGX** — game environments
- **Flashbax** — replay buffers
- **Scaling Scaling Laws** — experiments & model architecture
- **KataGo** — many methods
- Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)
