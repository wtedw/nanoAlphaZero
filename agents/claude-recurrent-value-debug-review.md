# Claude Opus 5.5 debugging consultation

2026-10-17 Asia/Tokyo. Requested explicitly by the user. Called the installed
Claude Code CLI with `--model claude-opus-5-5`, tools disabled, hooks disabled,
and no MCP servers. Successful response metadata reports `claude-opus-5-5`.
Input included the four investigation reports and production loss/unroll/mask
excerpts. Claude did not inspect the complete search implementation or run tests.

Exact prompt, response JSON (including model metadata), and extracted review:
`artifacts/claude-debug-consult-20261017/{prompt.txt,response.json,review.txt}`.
These ignored local artifacts preserve the full consultation. No training,
production changes, dependency changes, or commits were performed.

## Main recommendation

Before retraining, audit frozen checkpoints for:

1. Actual improved-policy entropy/max probability and Q-transformation scale
   relative to prior logits. Do not assume targets are one-hot.
2. Candidate-level recurrent/fresh-board/reference values for all considered
   actions at the same roots, aligned to one player's perspective. Measure
   ranking, prior-rank dependence, and selected-action optimism relative to
   the candidate average. Use fixed reference settings and paired uncertainty.
3. Whether recurrent predictions mostly preserve the root estimate with
   alternating sign instead of reflecting action-dependent successor changes.
4. Subsequently, inference-only search-scale/candidate-count comparisons,
   clearly separated from training changes. These assess exploitation of value
   noise but do not identify its training origin.

Claude's leading hypothesis combines limited supervision of alternative actions
with outcome targets that permit a root-outcome persistence shortcut. Search
could amplify optimistic errors and feed them back through sharp policy targets.
It favors this over strong policy/value gradient interference, given the previous
fixed-data experiment. This is a hypothesis ranking, not an established cause.

## Qualifications to the review

- Its claim that m=16/n=32 is effectively one-step search must be checked with
  actual visit/depth statistics. Simulation count alone does not establish that
  interior priors, legality, or deeper dynamics are irrelevant.
- Repeated signed outcome labels do not imply zero action-dependent value
  gradients. They permit shortcuts; whether the model uses them needs evidence.
- A regression on root and fresh successor values has correlated predictors.
  Coefficient magnitudes alone cannot establish or reject a shortcut. Compare
  held-out prediction gains and action sensitivity with appropriate controls.
- A successful held-out linear probe of latent states is evidence of recoverable
  information. A failed linear probe does not prove the dynamics lacks information;
  information can be nonlinear or the probe/data insufficient.
- No prior-rank trend does not by itself rule out action-coverage failure. Prior
  rank is only a proxy for historical action coverage.
- Played versus unplayed successor distillation is a potentially useful future
  comparison, but different benefits would not uniquely identify coverage: action
  difficulty, teacher errors, and target distributions also differ.
- Training arms inevitably change subsequent behavior, even if a target-only
  modification leaves the initial action-selection rule unchanged. Frozen-data
  comparisons isolate immediate gradient effects better than online runs.
- Claude's proposed long baseline runs, checkpoint cadence changes, sampler fix,
  and distillation arms are suggestions only. They have not been authorized by
  this consultation request or executed.

Recommended immediate next work: frozen target/search-depth audit plus the
candidate-value table; then test one-step versus accumulated dynamics error on
matched action sequences. Select a training intervention only after those results.

## Follow-up after the frozen probes

The user requested a second discussion of likely cause and fixes. Successful
Claude Code response again identifies `claude-opus-5-5`. Exact input and output
are `artifacts/claude-debug-consult-20261017/followup-prompt.txt`,
`followup-response.json`, and `followup-review.txt`. Tools remained disabled.

Claude revised its leading explanation to inaccurate action-dependent values
amplified by the Q transform into decisions and sharp targets. It explicitly
retracted the leading root-copy explanation, the low-prior-error prediction,
the effectively-one-step assertion, and assumed min-max rescaling. It demoted
counterfactual coverage to untested and acknowledged no causal onset evidence.

Its preferred first intervention is smaller `value_scale`, gated by a frozen
sweep on the latest cycle-3600 checkpoint. This is a mitigation of search's
sensitivity to inaccurate values, not a repair of dynamics. Select on one
position set and validate on a separate set; require improvement against both
current search and the prior-only comparator, rather than merely approaching
the prior as search becomes weaker. Claude suggested 0.05–0.3 as an exploratory
range, not a validated setting. No such sweep or setting change has been made.

To address the source of error, it recommends matched readout probes of learned
and actual successor representations, split by game and scored using within-root
rankings and decision quality. A monotone scalar calibration cannot repair wrong
rankings. Recovering ranking with a new head would support a readout problem;
failed recovery does not establish missing information without successful
real-state controls and adequate probe capacity/data. Distillation to fresh-board
values is a candidate subsequent intervention, not an approved or established fix.

Mixed completion remains enabled. With 32 root visits, the raw root estimate
has weight 1/33 in the mixed estimate used for unvisited actions. Visited actions
keep backed-up Q; mixed completion does not shrink each visited action toward
the root. Interior selection also uses the Q transform.

Our qualifications: the recurrent/fresh discrepancy is not a measured pure noise
standard deviation, so converting 0.25 into a proposed scale is only a heuristic.
The measured selected optimism subtracts an unweighted candidate mean, not the
prior-weighted mixed value. Claude's statement that target sharpness is largely
the winner's-curse gap is stronger than the evidence. A smaller scale affects
root/interior selection, targets, and subsequent behavior; a training comparison
would test that combined intervention, not isolate policy-target sharpness.
Because current checkpoints lack replay, a matched restart control is necessary
before attributing later training differences to any intervention.

No training, search sweep, dependency changes, or commits were performed during
this consultation. Current conclusion: demonstrated search-time amplification;
unresolved training origin. First validate a mitigation on the latest checkpoint,
and investigate readout versus dynamics before selecting an underlying value fix.
