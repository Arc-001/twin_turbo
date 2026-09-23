import numpy as np
import pandas as pd
import pytest

from src.decision.hybrid import EscalationMonitor, EscalationThresholds
from src.decision.policy import GROUND_NOW, SAFE, SCHEDULE_MAINTENANCE, WATCH, classify_zone
from src.decision.replay import age_outcome, optimal_age, unit_outcome
from src.eval.metrics import phm08_score
from src.twin.degradation import FleetPrior, UnitFit, fit_unit, hi_curve, time_to_threshold
from src.twin.particle_filter import ParticleTwin
from src.uncertainty.conformal import MondrianConformal, picp


def test_phm08_penalizes_late_more_than_early():
    y = np.array([50.0])
    assert phm08_score(y, y + 10) > phm08_score(y, y - 10)
    assert phm08_score(y, y) == 0.0


def test_classify_zone_boundaries():
    assert classify_zone(61) == SAFE
    assert classify_zone(60) == WATCH
    assert classify_zone(30) == SCHEDULE_MAINTENANCE
    assert classify_zone(10) == GROUND_NOW


def test_conformal_reaches_nominal_coverage():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 125, 20000)
    y = pred + rng.normal(0, 2 + pred / 10)  # heteroscedastic noise
    cal, test = slice(0, 10000), slice(10000, None)
    cp = MondrianConformal(alpha=0.1).fit(pred[cal], y[cal])
    lo, hi = cp.interval(pred[test])
    assert picp(y[test], lo, hi) >= 0.88
    # Mondrian bins adapt the width: low-RUL band must be narrower
    lo_small, hi_small = cp.interval(np.array([5.0]))
    lo_big, hi_big = cp.interval(np.array([110.0]))
    assert (hi_small - lo_small) < (hi_big - lo_big)


def test_time_to_threshold_inverts_curve():
    phi, theta, beta, h = 1.0, 0.02, 0.02, 0.0
    t = time_to_threshold(phi, theta, beta, h)
    assert hi_curve(np.array([t]), phi, theta, beta)[0] == pytest.approx(h, abs=1e-9)


def test_fit_unit_recovers_parameters():
    t = np.arange(1, 201, dtype=float)
    hi = hi_curve(t, 1.0, 0.02, 0.02) + np.random.default_rng(1).normal(0, 0.01, len(t))
    f = fit_unit(t, hi)
    assert f.beta == pytest.approx(0.02, rel=0.15)


def _synthetic_prior(n=60, seed=0):
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(n):
        phi, theta, beta = rng.normal(1, 0.02), np.exp(rng.normal(np.log(0.02), 0.2)), np.exp(rng.normal(np.log(0.02), 0.1))
        life = int(time_to_threshold(phi, theta, beta, 0.0))
        fits.append(UnitFit(phi, theta, beta, 0.0 + rng.normal(0, 0.02), 0.03, life))
    return FleetPrior.from_fits(fits)


def test_particle_twin_tracks_true_rul():
    prior = _synthetic_prior()
    phi, theta, beta = 1.0, 0.025, 0.021
    life = time_to_threshold(phi, theta, beta, 0.0)
    t = np.arange(1, int(life * 0.7), dtype=float)
    hi = hi_curve(t, phi, theta, beta) + np.random.default_rng(3).normal(0, 0.03, len(t))
    twin = ParticleTwin(prior, seed=0)
    twin.assimilate(t, hi)
    s = twin.rul_summary()
    true_rul = life - t[-1]
    assert s["q05"] <= true_rul <= s["q95"]


def test_lazy_assimilation_matches_eager_in_distribution():
    prior = _synthetic_prior()
    t = np.arange(1, 120, dtype=float)
    hi = hi_curve(t, 1.0, 0.02, 0.02)
    eager = ParticleTwin(prior, seed=0)
    for i in range(len(t)):
        eager.assimilate(t[i : i + 1], hi[i : i + 1])
    lazy = ParticleTwin(prior, seed=0)
    lazy.assimilate(t, hi)
    assert eager.rul_summary()["q50"] == pytest.approx(lazy.rul_summary()["q50"], rel=1e-6)


def test_escalation_monitor_triggers():
    m = EscalationMonitor(EscalationThresholds(watch_rul=60, delta_drop=8, resync_period=5))
    assert m.step(120) == ["periodic_resync"]  # first cycle forces a sync
    assert m.step(119) == []
    assert m.step(105) == ["sharp_drop"]
    steps = [m.step(104) for _ in range(5)]
    assert steps[:4] == [[]] * 4 and steps[4] == ["periodic_resync"]
    assert "low_rul" in m.step(50)


def test_unit_outcome_failure_and_waste():
    g = pd.DataFrame({"unit": 1, "cycle": np.arange(1, 11), "rul_true": np.arange(9, -1, -1, dtype=float)})
    never = unit_outcome(g, np.full(10, 100.0), np.zeros(10, bool), ground=5)
    assert never["failed"] and never["operated"] == 10
    early = unit_outcome(g, np.array([100, 100, 4, 0, 0, 0, 0, 0, 0, 0.0]), np.zeros(10, bool), ground=5)
    assert not early["failed"] and early["wasted"] == 7 and early["operated"] == 3


def test_optimal_age_is_before_shortest_life_when_failures_expensive():
    lifetimes = np.array([100, 101, 102, 103])
    a = optimal_age(lifetimes)
    assert a < 100
    assert not age_outcome(100, a)["failed"]
