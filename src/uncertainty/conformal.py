"""Split-conformal prediction intervals with Mondrian (binned) calibration.

Conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2021) wraps any
point regressor and returns intervals with a finite-sample coverage guarantee
of at least 1 - alpha, assuming exchangeability. Residual scale in RUL
prediction depends strongly on how far from failure an engine is, so the
calibration residuals are partitioned by predicted-RUL bin (Mondrian
conformal) and each bin gets its own quantile.
"""

from dataclasses import dataclass, field

import numpy as np

DEFAULT_BINS = (0.0, 30.0, 60.0, 90.0, np.inf)


@dataclass
class MondrianConformal:
    alpha: float = 0.1
    bin_edges: tuple[float, ...] = DEFAULT_BINS
    q_lo: np.ndarray = field(default=None, init=False)
    q_hi: np.ndarray = field(default=None, init=False)

    def _bin(self, pred: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(pred, self.bin_edges[1:-1]), 0, len(self.bin_edges) - 2)

    def fit(self, pred: np.ndarray, y: np.ndarray) -> "MondrianConformal":
        """Asymmetric (two one-sided) conformal: separate lower/upper residual
        quantiles, since RUL errors are skewed near the clip ceiling."""
        bins = self._bin(pred)
        resid = y - pred
        n_bins = len(self.bin_edges) - 1
        self.q_lo = np.zeros(n_bins)
        self.q_hi = np.zeros(n_bins)
        for b in range(n_bins):
            r = resid[bins == b]
            if len(r) == 0:
                r = resid
            n = len(r)
            level_lo = min(1.0, np.ceil((n + 1) * (1 - self.alpha / 2)) / n)
            self.q_lo[b] = np.quantile(-r, level_lo)
            self.q_hi[b] = np.quantile(r, level_lo)
        return self

    def interval(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = self._bin(np.asarray(pred))
        return pred - self.q_lo[b], pred + self.q_hi[b]


def picp(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Prediction-interval coverage probability."""
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lo: np.ndarray, hi: np.ndarray) -> float:
    """Mean prediction-interval width."""
    return float(np.mean(hi - lo))
