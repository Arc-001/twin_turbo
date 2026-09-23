"""Exponential degradation model for the health index.

    HI(t) = phi - theta * exp(beta * t)

phi   : initial health (units start with unknown manufacturing wear)
theta : damage scale at t = 0
beta  : damage growth rate

This is the classical stochastic exponential degradation model used for
residual-life prediction (Gebraeel et al., 2005). Failure happens when HI
crosses a unit-specific threshold h_fail, giving a closed-form time-to-failure:

    t_fail = ln((phi - h_fail) / theta) / beta
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def hi_curve(t: np.ndarray, phi: float, theta: float, beta: float) -> np.ndarray:
    return phi - theta * np.exp(beta * t)


def time_to_threshold(phi, theta, beta, h_fail):
    """Vectorized closed-form crossing time; returns 0 where already crossed."""
    ratio = np.maximum((phi - h_fail) / theta, 1.0)
    return np.log(ratio) / beta


@dataclass
class UnitFit:
    phi: float
    theta: float
    beta: float
    h_fail: float
    resid_std: float
    life: int


def fit_unit(t: np.ndarray, hi: np.ndarray) -> UnitFit | None:
    life = int(t[-1])
    p0 = (float(np.median(hi[:20])), 0.05, 3.0 / max(life, 1))
    try:
        popt, _ = curve_fit(
            hi_curve, t, hi, p0=p0,
            bounds=([-1.0, 1e-6, 1e-4], [3.0, 5.0, 0.5]),
            maxfev=20000,
        )
    except RuntimeError:
        return None
    phi, theta, beta = popt
    resid = hi - hi_curve(t, *popt)
    h_fail = float(hi_curve(np.array([life]), *popt)[0])
    return UnitFit(float(phi), float(theta), float(beta), h_fail, float(resid.std()), life)


@dataclass
class FleetPrior:
    """Population-level distribution of degradation parameters, learned from
    run-to-failure training engines. Parameterized as (phi, log theta, log beta)
    ~ multivariate normal, and h_fail ~ normal."""

    mean: np.ndarray
    cov: np.ndarray
    h_fail_mean: float
    h_fail_std: float
    obs_std: float
    fits: list[UnitFit]

    @classmethod
    def from_fits(cls, fits: list[UnitFit]) -> "FleetPrior":
        z = np.array([[f.phi, np.log(f.theta), np.log(f.beta)] for f in fits])
        h = np.array([f.h_fail for f in fits])
        return cls(
            mean=z.mean(axis=0),
            cov=np.cov(z, rowvar=False),
            h_fail_mean=float(h.mean()),
            h_fail_std=float(h.std()),
            obs_std=float(np.median([f.resid_std for f in fits])),
            fits=fits,
        )
