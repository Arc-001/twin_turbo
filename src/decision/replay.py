"""Replay maintenance policies over run-to-failure traces and score them.

Outcome model (per engine):
  - The first cycle whose action is GROUND_NOW removes the engine for
    preventive maintenance (cost C_p). Life left on the table = true RUL then.
  - If the engine reaches its failure cycle (true RUL = 0) first, it is an
    unplanned failure (cost C_f).
Fleet metric is the long-run cost rate from renewal-reward theory,
    cost_rate = total cost / total cycles operated,
the same objective that defines classical age-replacement (Barlow & Hunter, 1960).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.decision.hybrid import EscalationMonitor, EscalationThresholds
from src.decision.policy import DecisionThresholds

Z90 = 1.6449  # one-sided 95% normal quantile -> matches the 5%-quantile lower bounds


@dataclass(frozen=True)
class Costs:
    preventive: float = 1.0
    failure: float = 10.0


def hybrid_lower_bound(g: pd.DataFrame, escalation: EscalationThresholds, fusion: str = "min",
                       twin_lo: str = "twin_q05") -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Replay the escalation monitor over one engine's trace.

    Between syncs the edge conformal lower bound is used. At a sync:
      fusion="twin": trust the twin's posterior lower quantile alone
      fusion="min":  take the more cautious of the two calibrated bounds; each is
                     a one-sided 95% bound, so the minimum keeps >= 90% coverage
                     by the union bound while inheriting the edge model's
                     sharpness near failure and the twin's long-horizon tracking.
    """
    monitor = EscalationMonitor(escalation)
    edge, edge_lo, tlo = g["edge"].to_numpy(), g["edge_lo"].to_numpy(), g[twin_lo].to_numpy()
    lo = edge_lo.astype(float).copy()
    synced = np.zeros(len(g), dtype=bool)
    reasons_all = []
    for i in range(len(g)):
        reasons = monitor.step(float(edge[i]))
        reasons_all.append(reasons)
        if reasons:
            synced[i] = True
            lo[i] = tlo[i] if fusion == "twin" else min(tlo[i], edge_lo[i])
    return lo, synced, reasons_all


def lower_bound(g: pd.DataFrame, policy: str, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Returns (lower bound per cycle, twin-synced flag per cycle)."""
    none = np.zeros(len(g), dtype=bool)
    if policy == "edge_point":
        return g["edge"].to_numpy(), none
    if policy == "edge_conformal":
        return g["edge_lo"].to_numpy(), none
    if policy == "lstm_point":
        return g["lstm"].to_numpy(), none
    if policy == "lstm_mc":
        return (g["lstm_mc_mean"] - Z90 * g["lstm_mc_std"]).to_numpy(), none
    if policy == "twin_point":
        return g["twin_q50"].to_numpy(), ~none
    if policy == "twin_always":
        return g["twin_q05"].to_numpy(), ~none
    if policy == "oracle":
        return g["rul_true"].to_numpy(), none
    if policy.startswith("hybrid"):
        lo, synced, _ = hybrid_lower_bound(g, kw.get("escalation", EscalationThresholds()),
                                           kw.get("fusion", "min"), kw.get("twin_lo", "twin_q05"))
        return lo, synced
    raise ValueError(policy)


def unit_outcome(g: pd.DataFrame, lo: np.ndarray, synced: np.ndarray, ground: float) -> dict:
    life = int(g["cycle"].iloc[-1])
    hits = np.flatnonzero(lo <= ground)
    stop = int(hits[0]) if len(hits) else len(g) - 1
    rul_at_stop = float(g["rul_true"].iloc[stop])
    failed = rul_at_stop <= 0
    operated = life if failed else int(g["cycle"].iloc[stop])
    return {"unit": int(g["unit"].iloc[0]), "life": life, "operated": operated, "failed": bool(failed),
            "wasted": 0.0 if failed else rul_at_stop, "syncs": int(synced[: stop + 1].sum()),
            "cycles": stop + 1}


def age_outcome(life: int, age: int) -> dict:
    failed = life <= age
    return {"life": life, "operated": life if failed else age, "failed": bool(failed),
            "wasted": 0.0 if failed else float(life - age), "syncs": 0, "cycles": min(life, age)}


def optimal_age(lifetimes: np.ndarray, costs: Costs = Costs()) -> int:
    """Barlow-Hunter age replacement on the empirical lifetime distribution."""
    lifetimes = np.asarray(lifetimes)
    best_age, best_rate = None, np.inf
    for a in range(1, int(lifetimes.max()) + 1):
        failed = lifetimes <= a
        cost = costs.failure * failed.sum() + costs.preventive * (~failed).sum()
        rate = cost / np.minimum(lifetimes, a).sum()
        if rate < best_rate:
            best_age, best_rate = a, rate
    return best_age


def aggregate(outcomes: list[dict], costs: Costs = Costs()) -> dict:
    df = pd.DataFrame(outcomes)
    n_fail = int(df["failed"].sum())
    n_pm = len(df) - n_fail
    total_cost = costs.failure * n_fail + costs.preventive * n_pm
    return {
        "n_units": len(df),
        "failures": n_fail,
        "failure_rate": n_fail / len(df),
        "mean_wasted_life": float(df.loc[~df["failed"], "wasted"].mean()) if n_pm else 0.0,
        "life_utilization": float(df["operated"].sum() / df["life"].sum()),
        "cost_rate_x1000": 1000 * total_cost / float(df["operated"].sum()),
        "twin_sync_rate": float(df["syncs"].sum() / df["cycles"].sum()),
    }


def evaluate_policy(traces: pd.DataFrame, policy: str, thresholds: DecisionThresholds = DecisionThresholds(),
                    costs: Costs = Costs(), **kw) -> tuple[dict, list[dict]]:
    outcomes = []
    for _, g in traces.groupby("unit", sort=True):
        lo, synced = lower_bound(g, policy, **kw)
        outcomes.append(unit_outcome(g, lo, synced, thresholds.ground))
    return aggregate(outcomes, costs), outcomes


def evaluate_age_policy(traces: pd.DataFrame, ages_by_fold: dict[int, int], costs: Costs = Costs()) -> dict:
    lives = traces.groupby("unit").agg(life=("cycle", "max"), fold=("fold", "first"))
    outcomes = [age_outcome(int(r.life), ages_by_fold[int(r.fold)]) for r in lives.itertuples()]
    return aggregate(outcomes, costs)
