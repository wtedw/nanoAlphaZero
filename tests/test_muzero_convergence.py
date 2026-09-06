from nanoalphazero.research.muzero.convergence import OpeningCoverageGate


def result(search=27, policy=27, total=27, unscored=0):
    return {prefix + key: value
            for prefix, wins in (("hex_eval/", search), ("hex_eval_policy/", policy))
            for key, value in (("perfect_opening_wins", wins),
                               ("perfect_opening_total", total), ("unscored", unscored))}


def test_requires_repeated_complete_evaluations_of_both_modes():
    gate = OpeningCoverageGate(3)
    assert not gate.observe(result())
    assert not gate.observe({})
    assert gate.streak == 1
    assert not gate.observe(result())
    assert not gate.observe(result(policy=26))
    assert gate.streak == 0
    assert not gate.observe(result())
    assert not gate.observe(result())
    assert gate.observe(result())


def test_empty_unscored_missing_and_disabled_cannot_succeed():
    gate = OpeningCoverageGate(1)
    assert not gate.observe(result(total=0, search=0, policy=0))
    assert not gate.observe(result(unscored=1))
    assert not gate.observe({"hex_eval/perfect_opening_total": 27})
    assert not OpeningCoverageGate().observe(result())
