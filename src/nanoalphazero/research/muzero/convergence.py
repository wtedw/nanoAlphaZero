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
