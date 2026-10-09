"""Host-side strength ladder against random play and frozen past selves.

The factory captures fixed dependencies; callers own the explicit evolving state.
Ratings are relative ladder estimates, not externally calibrated Elo ratings.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.eval.matches import all_opening_actions, evaluate_vs


@dataclass(frozen=True)
class LadderState:
    anchor_params: object = None
    anchor_elo: float = 0.0
    anchor_cycle: int = 0
    rung: int = 0
    qualifying_deltas: tuple = ()
    best_random_score: float = 0.0
    elo_curve: tuple = (0.0,)
    elo_cycles: tuple = (0,)


def make_ladder(config, *, env, policy_fn, chart_fn=None, separate_sessions=False):
    """Return ladder_fn(state, cycle, params) -> (LadderState, metrics).

    policy_fn uses the common run_mcts_fn signature. The ladder owns scheduling,
    openings, color-swapped matches, independent RNG, and anchor promotion.
    A disabled/not-due call returns the same state and no metrics. State is not
    checkpointed: each host run starts from the random-opponent baseline.
    chart_fn optionally renders rating history; neither tracker is imported here.
    """
    config = dict(config)
    if separate_sessions:
        import uuid
        session = uuid.uuid4().hex[:8]
    period = config.get("eval_period", 100)
    promote_score = config.get("eval_promote_score", 0.75)
    patience = config.get("eval_promote_patience", 2)
    random_cap = config.get("eval_vs_random_max_elo", 0.0)
    openings = all_opening_actions(env, config, plies=config.get("eval_opening_plies", 1)) if period else None

    def ladder_fn(state, cycle, params):
        if not period or cycle % period:
            return state, {}
        print(f"\n--- Strength eval at cycle {cycle} "
              f"({len(openings)} openings x2 colors per opponent) ---")
        random_key, anchor_key = jax.random.split(jax.random.PRNGKey(1234 + cycle))
        score_random, delta_random = evaluate_vs(
            policy_fn, env, config, params, None,
            openings=openings, key=random_key, label="random (Elo 0)",
        )
        if random_cap is not None:
            delta_random = min(delta_random, float(random_cap))
        best_random = max(state.best_random_score, score_random)
        if state.rung > 0 and score_random < state.best_random_score - 0.15:
            print(f"  [eval] ⚠️  forgetting? vs-random score {score_random:.3f} is "
                  f"well below high-water {state.best_random_score:.3f}", flush=True)

        if state.anchor_params is None:
            score_anchor, delta_anchor = score_random, delta_random
        else:
            score_anchor, delta_anchor = evaluate_vs(
                policy_fn, env, config, params, state.anchor_params,
                openings=openings, key=anchor_key,
                label=f"anchor R{state.rung}@{state.anchor_cycle}",
            )

        live_elo = state.anchor_elo + float(delta_anchor)
        metrics = {
            "eval/score_vs_random": float(score_random),
            "eval/elo_vs_random": float(delta_random),
            "eval/score_vs_anchor": float(score_anchor),
            "eval/elo_vs_anchor": float(delta_anchor),
            "eval/ladder_elo": float(live_elo),
            "eval/rung": int(state.rung),
        }
        if separate_sessions:
            # A fresh anchor history has a new reference scale, even when the
            # tracker resumes the same run. Never join its curve to an old one.
            for old, new in (("ladder_elo", "elo"), ("rung", "rung"),
                             ("score_vs_anchor", "score_vs_anchor"), ("elo_vs_anchor", "elo_vs_anchor")):
                metrics[f"eval/ladder/{session}/{new}"] = metrics.pop(f"eval/{old}")
        print(f"  [eval] ladder Elo: {live_elo:+.0f}  (rung {state.rung} @ "
              f"{state.anchor_elo:+.0f}, +{float(delta_anchor):.0f} over anchor)", flush=True)
        qualifying = (state.qualifying_deltas + (float(delta_anchor),)
                      if score_anchor >= promote_score else ())
        updated = replace(
            state, best_random_score=best_random, qualifying_deltas=qualifying,
            elo_curve=state.elo_curve + (live_elo,), elo_cycles=state.elo_cycles + (cycle,),
        )
        if len(qualifying) >= patience and score_anchor >= promote_score:
            height = float(np.mean(qualifying))
            updated = replace(
                updated, anchor_elo=state.anchor_elo + height,
                anchor_params=jax.tree.map(jnp.copy, params), anchor_cycle=cycle,
                rung=state.rung + 1, qualifying_deltas=(),
            )
            print(f"  [eval] ⬆ PROMOTE to rung {updated.rung}: anchor now @ "
                  f"{updated.anchor_elo:+.0f} Elo (rung height +{height:.0f}, "
                  f"avg of {patience} evals)", flush=True)
        if chart_fn is not None and len(updated.elo_curve) >= 3:
            print(f"\n── ladder Elo over cycles {updated.elo_cycles[1]}..{updated.elo_cycles[-1]} ──",
                  flush=True)
            print(chart_fn({"total": updated.elo_curve}), flush=True)
            print(flush=True)
        return updated, metrics

    return ladder_fn, LadderState()
