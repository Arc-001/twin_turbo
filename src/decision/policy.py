"""Action-zone policy: maps a calibrated RUL lower bound to an autonomous action.

Decisions use the lower end of a (1 - alpha) prediction interval rather than a
point estimate, so a wider (less certain) interval pushes the action toward
caution. Interval sources are all calibrated: conformal bands for the edge
model, posterior quantiles for the particle-filter twin.
"""

from dataclasses import dataclass

SAFE = "SAFE"
WATCH = "WATCH"
SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"
GROUND_NOW = "GROUND_NOW"

ZONE_ORDER = [GROUND_NOW, SCHEDULE_MAINTENANCE, WATCH, SAFE]


@dataclass(frozen=True)
class DecisionThresholds:
    safe: float = 60.0    # lower bound above this: normal operation
    watch: float = 30.0   # at or below: plan a maintenance slot
    ground: float = 10.0  # at or below: stop operating now


def classify_zone(rul_lower: float, thresholds: DecisionThresholds = DecisionThresholds()) -> str:
    if rul_lower <= thresholds.ground:
        return GROUND_NOW
    if rul_lower <= thresholds.watch:
        return SCHEDULE_MAINTENANCE
    if rul_lower <= thresholds.safe:
        return WATCH
    return SAFE
