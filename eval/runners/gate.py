"""Regression gate: compare run metrics against a recorded baseline (seam S11).

Pure decision function per docs/02-eval-framework.md §8: any gate metric that
degrades more than the threshold (default 1% relative) FAILS the run; all
others pass. Baselines change only via explicit benchmark-report commits.

Assumption (documented): every metric compared here is HIGHER-IS-BETTER.
Latency/cost metrics get their own direction handling when they join the gate.

Edge rules:
- Metric in baseline but missing from current run -> FAIL (silent drop is not allowed).
- Metric only in current run -> ignored here; baselines grow via explicit commits.
- baseline <= 0: relative degradation is undefined; fail only if current < baseline.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

DEGRADATION_THRESHOLD = 0.01


@dataclass(frozen=True)
class GateDecision:
    status: str  # "pass" | "fail"
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def _metric_verdict(name: str, current: float, baseline: float, threshold: float) -> str | None:
    """Return a failure description, or None when the metric holds."""
    if current < baseline and baseline <= 0:
        return f"{name}: {current} < baseline {baseline} (absolute comparison, baseline <= 0)"
    if baseline > 0:
        degradation = (baseline - current) / baseline
        # epsilon guards float noise at the exact threshold boundary
        if degradation > threshold + 1e-12:
            return (
                f"{name}: {current} vs baseline {baseline} "
                f"({degradation:.2%} relative loss > {threshold:.2%})"
            )
    return None


def compare_to_baseline(
    current_metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    threshold: float = DEGRADATION_THRESHOLD,
) -> GateDecision:
    """Decide PASS/FAIL of a run against the recorded baseline."""
    failures: list[str] = []
    for name, base_value in sorted(baseline.items()):
        if name not in current_metrics:
            failures.append(f"{name}: missing from current run (baseline {base_value})")
            continue
        verdict = _metric_verdict(name, current_metrics[name], base_value, threshold)
        if verdict is not None:
            failures.append(verdict)
    return GateDecision(status="fail" if failures else "pass", failures=failures)
