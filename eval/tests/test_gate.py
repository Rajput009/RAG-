"""Seam S11 groundwork: regression gate comparator vs hand-worked baselines."""

from eval.runners import DEGRADATION_THRESHOLD, GateDecision, compare_to_baseline

BASELINE = {"recall@5": 0.90, "recall@10": 0.95, "mrr@10": 0.90, "ndcg@10": 0.90}


class TestPassCases:
    def test_identical_metrics_pass(self) -> None:
        decision = compare_to_baseline(BASELINE, BASELINE)
        assert decision.passed
        assert decision.failures == []

    def test_improvements_pass(self) -> None:
        improved = {k: min(1.0, v + 0.05) for k, v in BASELINE.items()}
        assert compare_to_baseline(improved, BASELINE).passed

    def test_exactly_one_percent_degradation_passes(self) -> None:
        # gate is strictly-greater-than: exactly at threshold holds
        current = {"recall@5": 0.90 * (1 - DEGRADATION_THRESHOLD)}
        decision = compare_to_baseline(current, {"recall@5": 0.90})
        assert decision.passed


class TestFailCases:
    def test_slight_over_threshold_degradation_fails(self) -> None:
        current = {"recall@5": 0.90 * (1 - DEGRADATION_THRESHOLD - 0.0001)}
        decision = compare_to_baseline(current, {"recall@5": 0.90})
        assert not decision.passed
        assert "recall@5" in decision.failures[0]

    def test_missing_metric_from_current_run_fails(self) -> None:
        decision = compare_to_baseline({"recall@5": 0.9}, BASELINE)
        assert not decision.passed
        assert any("missing from current run" in f for f in decision.failures)

    def test_multiple_failures_all_reported(self) -> None:
        current = {"recall@5": 0.50, "recall@10": 0.50, "mrr@10": 0.89, "ndcg@10": 0.90}
        decision = compare_to_baseline(current, BASELINE)
        assert not decision.passed
        assert len(decision.failures) == 3  # recall@5, recall@10, mrr@10


class TestEdgeRules:
    def test_new_current_only_metrics_are_ignored(self) -> None:
        current = {**BASELINE, "citation_precision": 0.98}
        assert compare_to_baseline(current, BASELINE).passed

    def test_zero_baseline_uses_absolute_comparison(self) -> None:
        # baseline 0 cannot degrade relatively; fail only if strictly worse
        assert compare_to_baseline({"new_metric": 0.0}, {"new_metric": 0.0}).passed
        assert not compare_to_baseline({"weird": -0.1}, {"weird": 0.0}).passed

    def test_custom_threshold(self) -> None:
        current = {"m": 0.94}  # 6% loss
        assert compare_to_baseline(current, {"m": 1.0}, threshold=0.10).passed
        assert not compare_to_baseline(current, {"m": 1.0}, threshold=0.05).passed

    def test_decision_is_immutable(self) -> None:
        import dataclasses

        decision = GateDecision(status="pass")
        try:
            decision.status = "fail"  # type: ignore[misc]
            raised = False
        except dataclasses.FrozenInstanceError:
            raised = True
        assert raised
