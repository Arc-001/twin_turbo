"""Hybrid edge/twin fusion: cheap edge inference every cycle, twin sync on trigger.

Trigger conditions (any one escalates to the twin):
  - low_rul:          edge estimate has left the "clearly safe" band
  - sharp_drop:       edge estimate fell sharply vs the previous cycle (fault onset)
  - periodic_resync:  too many cycles since the twin last synchronized

Between syncs the edge device keeps a buffer of health-index observations (one
float per cycle). A sync hands that buffer to the engine's particle-filter twin,
which assimilates it and returns a full RUL posterior. At a sync the decision
uses the more cautious of the twin's 5% posterior quantile and the edge
conformal lower bound (min-fusion); without a sync it falls back to the edge
estimate with its conformal band.
"""

from dataclasses import dataclass, field

import numpy as np

from src.decision.policy import DecisionThresholds, classify_zone
from src.models.edge_model import EdgeRULModel
from src.models.features import summarize_windows
from src.twin.health_index import HealthIndexModel
from src.twin.particle_filter import ParticleTwin
from src.uncertainty.conformal import MondrianConformal

TRIGGERS = ("low_rul", "sharp_drop", "periodic_resync")


@dataclass(frozen=True)
class EscalationThresholds:
    watch_rul: float = 60.0
    delta_drop: float = 8.0
    resync_period: int = 20
    enabled: tuple[str, ...] = TRIGGERS


@dataclass
class EscalationMonitor:
    """Stateful trigger logic, shared by the live estimator and offline replay."""

    thresholds: EscalationThresholds = field(default_factory=EscalationThresholds)

    def __post_init__(self) -> None:
        self._last_edge: float | None = None
        self._since_sync = self.thresholds.resync_period  # force a sync on the first cycle

    def step(self, edge_rul: float) -> list[str]:
        th = self.thresholds
        self._since_sync += 1
        reasons = []
        if "low_rul" in th.enabled and edge_rul <= th.watch_rul:
            reasons.append("low_rul")
        if "sharp_drop" in th.enabled and self._last_edge is not None and self._last_edge - edge_rul >= th.delta_drop:
            reasons.append("sharp_drop")
        if "periodic_resync" in th.enabled and self._since_sync >= th.resync_period:
            reasons.append("periodic_resync")
        if self._last_edge is None and not reasons:
            reasons.append("initial_sync")
        if reasons:
            self._since_sync = 0
        self._last_edge = edge_rul
        return reasons


@dataclass
class HybridRULEstimator:
    """Live, cycle-by-cycle estimator for one engine."""

    edge_model: EdgeRULModel
    edge_conformal: MondrianConformal
    hi_model: HealthIndexModel
    twin: ParticleTwin
    escalation: EscalationThresholds = field(default_factory=EscalationThresholds)
    thresholds: DecisionThresholds = field(default_factory=DecisionThresholds)
    feature_cols: list[str] | None = None
    fusion: str = "min"  # "min" | "twin" -- see src/decision/replay.hybrid_lower_bound

    def __post_init__(self) -> None:
        self.monitor = EscalationMonitor(self.escalation)
        self._buffer_t: list[float] = []
        self._buffer_hi: list[float] = []
        self._sensor_idx = None
        if self.feature_cols is not None:
            self._sensor_idx = [self.feature_cols.index(c) for c in self.hi_model.sensor_cols]

    def step(self, window: np.ndarray, cycle: int) -> dict:
        """window: (window_size, n_features) normalized, oldest -> newest."""
        edge_rul = float(self.edge_model.predict(summarize_windows(window[None]))[0])
        edge_lo, edge_hi = (float(v[0]) for v in self.edge_conformal.interval(np.array([edge_rul])))

        hi_now = float(self.hi_model.transform_array(window[-1, self._sensor_idx]))
        self._buffer_t.append(float(cycle))
        self._buffer_hi.append(hi_now)

        reasons = self.monitor.step(edge_rul)
        out = {"cycle": cycle, "edge_rul": edge_rul, "edge_lo": edge_lo, "edge_hi": edge_hi,
               "hi": hi_now, "twin_synced": bool(reasons), "trigger_reasons": reasons}
        if reasons:
            self.twin.assimilate(np.array(self._buffer_t), np.array(self._buffer_hi))
            self._buffer_t.clear()
            self._buffer_hi.clear()
            s = self.twin.rul_summary()
            lo = s["q05"] if self.fusion == "twin" else min(s["q05"], edge_lo)
            out.update(final_rul=s["q50"], final_lo=lo, final_hi=s["q95"], source="twin",
                       twin_q05=s["q05"], twin_mean=s["mean"])
        else:
            out.update(final_rul=edge_rul, final_lo=edge_lo, final_hi=edge_hi, source="edge")
        out["zone"] = classify_zone(out["final_lo"], self.thresholds)
        return out
