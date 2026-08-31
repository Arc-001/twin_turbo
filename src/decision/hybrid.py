"""Hybrid edge/twin fusion: cheap edge inference every cycle, twin resync on trigger.

Trigger conditions (any one escalates to the twin):
  - low_rul:          edge estimate has left the "clearly safe" band
  - sharp_drop:       edge estimate fell sharply vs the previous cycle (fault onset)
  - periodic_resync:  too many cycles since the twin last ran (catches slow drift
                       the edge model's point estimate alone can mask)

When the twin runs, its MC-dropout mean/std becomes the estimate fed to the
decision policy (higher fidelity + a real uncertainty band). Otherwise the
edge point estimate is used with zero uncertainty -- the policy has no basis
to be cautious without the twin, which is exactly why "low_rul" escalates it.
"""

from dataclasses import dataclass, field

import numpy as np

from src.decision.policy import DecisionThresholds, classify_zone
from src.models.edge_model import EdgeRULModel
from src.models.features import summarize_windows
from src.models.twin_model import TwinRULModel


@dataclass(frozen=True)
class EscalationThresholds:
    watch_rul: float = 60.0
    delta_drop: float = 8.0
    resync_period: int = 20
    mc_samples: int = 20


@dataclass
class HybridRULEstimator:
    edge_model: EdgeRULModel
    twin_model: TwinRULModel
    escalation: EscalationThresholds = field(default_factory=EscalationThresholds)
    thresholds: DecisionThresholds = field(default_factory=DecisionThresholds)

    def __post_init__(self) -> None:
        self._last_edge_rul: float | None = None
        self._cycles_since_sync = self.escalation.resync_period  # force a sync on cycle 1

    def step(self, window: np.ndarray) -> dict:
        """window: (window_size, n_features), most recent cycles first->last."""
        edge_feat = summarize_windows(window[None, :, :])
        edge_rul = float(self.edge_model.predict(edge_feat)[0])

        delta = None if self._last_edge_rul is None else edge_rul - self._last_edge_rul
        self._cycles_since_sync += 1

        reasons = []
        if edge_rul <= self.escalation.watch_rul:
            reasons.append("low_rul")
        if delta is not None and delta <= -self.escalation.delta_drop:
            reasons.append("sharp_drop")
        if self._cycles_since_sync >= self.escalation.resync_period:
            reasons.append("periodic_resync")

        twin_triggered = len(reasons) > 0
        if twin_triggered:
            mean, std = self.twin_model.predict_with_uncertainty(
                window[None, :, :], n_samples=self.escalation.mc_samples
            )
            final_rul, final_std = float(mean[0]), float(std[0])
            self._cycles_since_sync = 0
        else:
            final_rul, final_std = edge_rul, 0.0

        zone = classify_zone(final_rul, final_std, self.thresholds)
        self._last_edge_rul = edge_rul

        return {
            "edge_rul": edge_rul,
            "twin_triggered": twin_triggered,
            "trigger_reasons": reasons,
            "final_rul": final_rul,
            "final_std": final_std,
            "zone": zone,
        }
