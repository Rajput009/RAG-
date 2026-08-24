"""Eval runners: dataset scoring, gating, security suites (docs/02 §5-§8)."""

from eval.runners.gate import DEGRADATION_THRESHOLD, GateDecision, compare_to_baseline

__all__ = ["DEGRADATION_THRESHOLD", "GateDecision", "compare_to_baseline"]
