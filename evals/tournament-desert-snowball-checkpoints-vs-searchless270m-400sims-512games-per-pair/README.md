# Desert Snowball vs Searchless Chess 270M

This tournament compares two released Desert Snowball checkpoints against the
Searchless Chess 270M model. Each checkpoint played 512 games at 400 MCTS
simulations, using paired openings with both player assignments.

## Model

Both checkpoints use the same 7.28M-parameter (`b10c256nbt`) KataGo-style
network. It has a 256-channel convolutional trunk with 10 nested-bottleneck
residual blocks, 128-channel bottlenecks, Mish activations, global-pooling
blocks, and RVGL 3x3/1x1 convolution branches. A chess policy head predicts
the 64x73 move encoding, while a WDL value head predicts win, draw, and loss.
Each float32 safetensors checkpoint is approximately 29.1 MB.

model34400 was trained in under 24 hours on a TPU v4-32 pod. model68800
represents 48 hours of training on the same hardware.

## Results

| Model | Wins | Draws | Losses | Score | Elo vs 270M |
| --- | ---: | ---: | ---: | ---: | ---: |
| model34400 | 205 | 144 | 163 | 54.1% | +27 |
| model68800 | 261 | 131 | 120 | 63.8% | +99 |

BayesElo estimates are relative to the 270M opponent. They show that both
checkpoints performed better than 270M in this tournament, with model68800
showing the stronger result. They should not be interpreted as absolute human
Elo ratings.

All 1,024 games completed with no failures. The full output is in
[`runs/20260818-121855-0a890a3592`](runs/20260818-121855-0a890a3592), including
the PGN, configuration, summary, and BayesElo output.
