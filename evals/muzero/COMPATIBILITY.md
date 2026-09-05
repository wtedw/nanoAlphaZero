# Environment compatibility

All environments use the existing `core.make_env` wrapper and `observe(state,
current_player)`. The wrapper batches PGX1 init/step/observe. All current games
alternate players 0/1 and return two-element, zero-sum rewards. Legal actions
are checked only at real roots. Illegal real actions fail the research run.

| Name | Observation | Actions | Legality | Full MuZero cycle/evaluation | Strength evidence |
|---|---|---:|---|---|---|
| ttt | 3×3×4 bool | 9 | empty squares | full-cap CPU train/save/load/evaluation | none |
| connect4 | 6×7×3 bool | 7 | nonfull columns | full-cap CPU train/save/load/evaluation | none |
| hex4 | 4×4×4 bool | 16 | empty squares | four-TPU self-play/replay/train/save and real-game evaluation | preliminary three-seed study |
| hex5 | 5×5×4 bool | 25 | empty squares | vector CPU full-cap cycle; spatial four-TPU staged training, train batch 4096 | one seed: 13/13 known winning MoHex openings in both modes; broad optimality unproven |
| hex6 | 6×6×4 bool | 36 | empty squares | full-cap CPU train/save/load/evaluation | none |
| hex7 | 7×7×4 bool | 49 | empty squares | full-cap CPU train/save/load/evaluation | none |
| hex8 | 8×8×4 bool | 64 | empty squares | full-cap CPU train/save/load/evaluation | none |
| hex9 | 9×9×4 bool | 81 | empty squares | full-cap CPU train/save/load/evaluation | none |
| go3 | 3×3×17 bool | 10 | board legality + pass | full-cap CPU train/save/load/evaluation | none |
| go4 | 4×4×17 bool | 17 | board legality + pass | full-cap CPU train/save/load/evaluation | none |
| go5 | 5×5×17 bool | 26 | board legality + pass | full-cap CPU train/save/load/evaluation | none |
| go6 | 6×6×17 bool | 37 | board legality + pass | full-cap CPU train/save/load/evaluation | none |
| go7 | 7×7×17 bool | 50 | board legality + pass | full-cap CPU train/save/load/evaluation | none |
| go8 | 8×8×17 bool | 65 | board legality + pass | full-cap CPU train/save/load/evaluation | none |
| go9 | 9×9×17 bool | 82 | board legality + pass | full-cap CPU train/save/load/evaluation | none |
| chess | 8×8×119 float32 | 4672 | packed 146-word uint32 mask | full-cap CPU train/save/load/evaluation | none |

Full-cap smoke means legal collection through the configured game horizon,
finite aligned targets, an optimizer update, checkpoint round trip, and real
evaluation games after loading. All 16 pass this CPU compatibility gate with
a small **vector** model, now including production consume, staged unroll drain,
position replay, and staged-state checkpoint loading (16 tests, 337.88 seconds).
Spatial architecture compatibility has been exercised on Hex4 and Hex5, including
full-size Hex5 four-TPU staged smoke. The action encoders also have focused shape/orientation tests;
that is not all-environment spatial training validation.
Production-size TPU memory and speed remain unvalidated except
for Hex 4 and Hex 5 (Hex5 smoke uses reduced replay capacity; the completed
seed-0 training run uses full replay capacity). Terminal/absorbing/return semantics have separate controlled tests.
Chess games can reach the rollout cap and then train with a bootstrap; evaluation
reports capped games as draws. No full-game termination rate or strength is
claimed from these four-game compatibility samples.

Terminal rewards are ±1, or zero for draws in TTT/Connect4/chess. Hex has no
swap and no draws. Go uses PGX1's size-specific half-integer komi and its
termination rules (passes, repetition and move cap); it returns the final
winner, not territory margin. Go's observation contains history. Chess includes
history/rule counters in its 119 planes. Representation currently consumes
the observation provided by the production wrapper; no additional history is
assembled by MuZero. Simulator truncation and the research rollout cap both
retain a boundary bootstrap and do not silently become draws during training.

The model uses bounded scalar reward/value heads suited to these games. General
unbounded rewards, chance nodes, extra turns, multiplayer and nonzero-sum games
are outside the tested contract. They are not production-supported environments.
