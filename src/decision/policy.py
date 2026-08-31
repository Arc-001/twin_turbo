"""Action-zone policy: maps a (RUL estimate, uncertainty) pair to an autonomous action.

Uses a lower-confidence-bound (RUL - k*std) for zone classification rather than
the raw point estimate, so higher twin uncertainty pushes the decision toward
caution -- standard practice for safety-critical thresholding under uncertainty.
"""

from dataclasses import dataclass

SAFE = "SAFE"
WATCH = "WATCH"
SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"
GROUND_NOW = "GROUND_NOW"

ZONE_ORDER = [GROUND_NOW, SCHEDULE_MAINTENANCE, WATCH, SAFE]


@dataclass(frozen=True)
class DecisionThresholds:
    safe: float = 60.0    # effective RUL above this: normal operation
    watch: float = 30.0   # below this: elevated monitoring
    ground: float = 10.0  # below this: stop operation now
    uncertainty_k: float = 1.0  # confidence margin multiplier applied to std


def classify_zone(rul_estimate: float, rul_std: float, thresholds: DecisionThresholds = DecisionThresholds()) -> str:
    effective_rul = rul_estimate - thresholds.uncertainty_k * rul_std
    if effective_rul <= thresholds.ground:
        return GROUND_NOW
    if effective_rul <= thresholds.watch:
        return SCHEDULE_MAINTENANCE
    if effective_rul <= thresholds.safe:
        return WATCH
    return SAFE
