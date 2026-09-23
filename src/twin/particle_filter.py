"""Per-engine digital twin: a particle filter over degradation parameters.

Each engine gets its own twin instance. Particles are hypotheses about that
engine's (phi, theta, beta, h_fail). Every assimilated health-index observation
reweights them; the surviving cloud is the twin's belief about *this* engine,
and pushing each particle forward to its failure threshold yields a full RUL
distribution (not just a point estimate).

Assimilation can be lazy: the edge device buffers cheap HI observations and the
twin ingests the whole buffer when it is synchronized, so twin cost is paid
only on escalation.
"""

from dataclasses import dataclass, field

import numpy as np

from src.twin.degradation import FleetPrior, hi_curve, time_to_threshold


@dataclass
class ParticleTwin:
    prior: FleetPrior
    n_particles: int = 2000
    obs_std_scale: float = 1.0
    shrink: float = 0.98  # Liu-West kernel shrinkage for static-parameter jitter
    seed: int = 0
    _t_seen: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        z = self.rng.multivariate_normal(self.prior.mean, self.prior.cov, size=self.n_particles)
        h = self.rng.normal(self.prior.h_fail_mean, self.prior.h_fail_std, size=self.n_particles)
        self.z = np.column_stack([z, h])  # phi, log theta, log beta, h_fail
        self.logw = np.zeros(self.n_particles)
        self.sigma = self.prior.obs_std * self.obs_std_scale

    @property
    def params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.z[:, 0], np.exp(self.z[:, 1]), np.exp(self.z[:, 2]), self.z[:, 3]

    @property
    def weights(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def ess(self) -> float:
        w = self.weights
        return float(1.0 / np.sum(w**2))

    def assimilate(self, t: np.ndarray, hi: np.ndarray) -> None:
        """Ingest a batch of (cycle, HI) observations."""
        for ti, hi_i in zip(np.atleast_1d(t), np.atleast_1d(hi)):
            phi, theta, beta, _ = self.params
            pred = hi_curve(ti, phi, theta, beta)
            self.logw += -0.5 * ((hi_i - pred) / self.sigma) ** 2
            if self.ess() < self.n_particles / 2:
                self._resample()
            self._t_seen = int(ti)

    def _resample(self) -> None:
        w = self.weights
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        idx = np.searchsorted(np.cumsum(w), positions)
        idx = np.minimum(idx, self.n_particles - 1)
        z = self.z[idx]
        # Liu-West: shrink toward the mean, add matched-variance jitter
        mean = z.mean(axis=0)
        cov = np.cov(z, rowvar=False) + 1e-9 * np.eye(z.shape[1])
        a = self.shrink
        h2 = 1 - a**2
        z = a * z + (1 - a) * mean + self.rng.multivariate_normal(np.zeros(z.shape[1]), h2 * cov, size=len(z))
        self.z = z
        self.logw = np.zeros(self.n_particles)

    def rul_samples(self, t_now: int | None = None, n: int = 2000) -> np.ndarray:
        t_now = self._t_seen if t_now is None else t_now
        phi, theta, beta, h_fail = self.params
        rul = np.maximum(time_to_threshold(phi, theta, beta, h_fail) - t_now, 0.0)
        idx = self.rng.choice(self.n_particles, size=n, p=self.weights)
        return rul[idx]

    def hi_forecast(self, t_grid: np.ndarray, quantiles=(0.05, 0.5, 0.95), n: int = 400) -> np.ndarray:
        """Projected HI bands over future cycles -- the twin's what-if trajectory."""
        idx = self.rng.choice(self.n_particles, size=n, p=self.weights)
        phi, theta, beta, _ = (p[idx] for p in self.params)
        curves = hi_curve(t_grid[None, :], phi[:, None], theta[:, None], beta[:, None])
        return np.quantile(curves, quantiles, axis=0)

    def rul_particles(self, t_now: int | None = None) -> np.ndarray:
        t_now = self._t_seen if t_now is None else t_now
        phi, theta, beta, h_fail = self.params
        return np.maximum(time_to_threshold(phi, theta, beta, h_fail) - t_now, 0.0)

    def rul_summary(self, quantiles=(0.05, 0.5, 0.95), clip: float | None = None) -> dict[str, float]:
        """Posterior RUL mean + weighted quantiles (no resampling noise)."""
        rul = self.rul_particles()
        if clip is not None:
            rul = np.minimum(rul, clip)
        w = self.weights
        order = np.argsort(rul)
        cdf = np.cumsum(w[order])
        qs = [float(rul[order][min(np.searchsorted(cdf, q), len(rul) - 1)]) for q in quantiles]
        mean = float(np.sum(w * rul))
        std = float(np.sqrt(np.sum(w * (rul - mean) ** 2)))
        return {"mean": mean, "std": std, **{f"q{int(q * 100):02d}": v for q, v in zip(quantiles, qs)}}
