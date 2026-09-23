#!/usr/bin/env python3
"""Turn cached traces into every table, sweep and ablation in the report.

Reads artifacts/traces/*, writes artifacts/results/results.json.

Usage:
    python scripts/evaluate.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.decision.hybrid import TRIGGERS, EscalationThresholds
from src.decision.policy import DecisionThresholds
from src.decision.replay import Z90, Costs, aggregate, age_outcome, evaluate_age_policy, evaluate_policy
from src.eval.literature import LITERATURE_RMSE
from src.eval.metrics import phm08_score, rmse
from src.models.features import summarize_windows
from src.models.lstm_model import LSTMRULModel
from src.twin.particle_filter import ParticleTwin
from src.uncertainty.conformal import mpiw, picp

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
CLIP = 125.0
POLICIES = ["oracle", "edge_point", "edge_conformal", "lstm_point", "lstm_mc", "twin_point", "twin_always", "hybrid"]
SWEEP_GROUND = [0, 2, 5, 10, 15, 20, 25, 30]
HORIZON_BINS = [0, 25, 50, 75, 100, 126]


def intervals(t: pd.DataFrame) -> dict[str, tuple]:
    return {
        "edge_conformal": (t["edge_lo"], t["edge_hi"]),
        "lstm_mc_dropout": (t["lstm_mc_mean"] - Z90 * t["lstm_mc_std"], t["lstm_mc_mean"] + Z90 * t["lstm_mc_std"]),
        "twin_posterior": (np.minimum(t["twin_q05"], CLIP), np.minimum(t["twin_q95"], CLIP)),
    }


def benchmark(tr_dir: Path, ds: str) -> dict:
    t = pd.read_pickle(tr_dir / f"{ds}_test.pkl")
    g = pd.read_pickle(tr_dir / f"{ds}_test_globalnorm.pkl")
    y = t["rul"].to_numpy()
    out = {"n_units": len(t), "accuracy": {}, "coverage": {}, "global_norm": {}}
    for m in ["edge", "lstm", "twin"]:
        out["accuracy"][m] = {"rmse": rmse(y, t[m]), "phm08": phm08_score(y, t[m]),
                              "mae": float(np.mean(np.abs(y - t[m])))}
    for m in ["edge", "twin"]:
        out["global_norm"][m] = {"rmse": rmse(y, g[m]), "phm08": phm08_score(y, g[m])}
    for name, (lo, hi) in intervals(t).items():
        out["coverage"][name] = {"picp": picp(y, lo, hi), "mpiw": mpiw(lo, hi)}
    return out


def cv_accuracy(cv: pd.DataFrame) -> dict:
    y = cv["rul"].to_numpy()
    out = {"overall": {m: rmse(y, cv[m]) for m in ["edge", "lstm", "twin"]}, "by_horizon": [],
           "coverage": {name: {"picp": picp(y, lo, hi), "mpiw": mpiw(lo, hi)}
                        for name, (lo, hi) in intervals(cv).items()}}
    bins = pd.cut(cv["rul_true"], HORIZON_BINS, right=False)
    for b, grp in cv.groupby(bins, observed=True):
        yy = grp["rul"].to_numpy()
        row = {"lo": int(b.left), "hi": int(b.right), "n": len(grp)}
        row.update({m: rmse(yy, grp[m]) for m in ["edge", "lstm", "twin"]})
        for name, (lo, hi) in intervals(grp).items():
            row[f"picp_{name}"] = picp(yy, lo, hi)
        out["by_horizon"].append(row)
    return out


def decisions(cv: pd.DataFrame, ages: dict[int, int]) -> dict:
    out = {"default": {}, "sweep": [], "triggers": [], "resync": [], "fusion": {}}
    out["default"]["age_replacement"] = evaluate_age_policy(cv, ages)
    lives = cv.groupby("unit")["cycle"].max().to_numpy()
    out["default"]["run_to_failure"] = aggregate([age_outcome(int(l), 10**6) for l in lives])
    for p in POLICIES:
        out["default"][p] = evaluate_policy(cv, p)[0]

    for ground in SWEEP_GROUND:
        th = DecisionThresholds(ground=ground)
        for p in ["edge_point", "edge_conformal", "lstm_point", "lstm_mc", "twin_point", "twin_always", "hybrid"]:
            out["sweep"].append({"ground": ground, "policy": p, **evaluate_policy(cv, p, th)[0]})

    # trigger ablation: all, each removed, each alone
    variants = {"all": TRIGGERS}
    variants.update({f"without_{t}": tuple(x for x in TRIGGERS if x != t) for t in TRIGGERS})
    variants.update({f"only_{t}": (t,) for t in TRIGGERS})
    for ground in (10, 2):
        th = DecisionThresholds(ground=ground)
        for name, enabled in variants.items():
            esc = EscalationThresholds(enabled=enabled)
            out["triggers"].append({"variant": name, "ground": ground,
                                    **evaluate_policy(cv, "hybrid", th, escalation=esc)[0]})
    for period in [5, 10, 20, 40, 80, 10**6]:
        esc = EscalationThresholds(resync_period=period)
        out["resync"].append({"period": period, **evaluate_policy(cv, "hybrid", DecisionThresholds(ground=2),
                                                                   escalation=esc)[0]})
    for ground in (10, 2, 0):
        th = DecisionThresholds(ground=ground)
        out["fusion"][str(ground)] = {f: evaluate_policy(cv, "hybrid", th, fusion=f)[0] for f in ("min", "twin")}
    # Safety/efficiency frontier: for each policy, the cheapest threshold that
    # produced zero failures -- how aggressive can each policy afford to be?
    sweep = pd.DataFrame(out["sweep"])
    out["safe_operating_point"] = {}
    for p, grp in sweep.groupby("policy"):
        safe = grp[grp["failures"] == 0]
        if len(safe):
            best = safe.loc[safe["cost_rate_x1000"].idxmin()]
            out["safe_operating_point"][p] = {"ground": int(best["ground"]), "cost_rate_x1000": float(best["cost_rate_x1000"]),
                                              "mean_wasted_life": float(best["mean_wasted_life"]),
                                              "min_safe_ground": int(safe["ground"].min())}
        else:
            out["safe_operating_point"][p] = None
    return out


def op_counts(bundle, lstm_params: int, n_feat: int, window: int = 30) -> dict:
    """Hardware-independent per-inference work, so the edge/twin split is not
    judged by Python interpreter overhead on a laptop."""
    trees = bundle.edge.model._predictors
    n_nodes = sum(len(t[0].nodes) for t in trees)
    depth = max(int(t[0].nodes["depth"].max()) for t in trees)
    h = bundle.lstm.hidden_size if bundle.lstm is not None else 64
    lstm_macs = window * 4 * h * (n_feat + h + 1) + h
    n_p = 2000
    return {
        "edge_trees": len(trees), "edge_total_nodes": int(n_nodes),
        "edge_comparisons_per_inference": int(len(trees) * depth),
        "edge_feature_flops": int(window * n_feat * 8),
        "lstm_macs_per_pass": int(lstm_macs), "lstm_mc20_macs": int(20 * lstm_macs),
        "twin_state_floats_per_engine": n_p * 4,
        "twin_flops_per_observation": int(n_p * 12),
    }


def latency(art: Path, ds: str) -> dict:
    with open(art / "models" / f"{ds}_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    n_feat = len(bundle.data.feature_cols)
    rng = np.random.default_rng(0)
    window = rng.normal(size=(1, 30, n_feat))
    feat = summarize_windows(window)

    def bench(fn, reps=200):
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        return {"mean_ms": float(np.mean(ts)), "p95_ms": float(np.percentile(ts, 95))}

    out = {"edge_predict": bench(lambda: bundle.edge.predict(summarize_windows(window))),
           "edge_model_bytes": len(pickle.dumps(bundle.edge))}
    twin = ParticleTwin(bundle.prior)
    obs_t, obs_hi = np.arange(1, 21, dtype=float), np.linspace(0.9, 0.8, 20)
    out["twin_sync_20obs"] = bench(lambda: (ParticleTwin(bundle.prior).assimilate(obs_t, obs_hi),), reps=30)
    out["twin_particles"] = twin.n_particles
    lstm_cpu = LSTMRULModel.load(str(art / "models" / f"{ds}_lstm.pt"), device="cpu")
    out["lstm_cpu_predict"] = bench(lambda: lstm_cpu.predict(window))
    out["lstm_cpu_mc20"] = bench(lambda: lstm_cpu.predict_with_uncertainty(window, 20), reps=50)
    out["lstm_params"] = int(sum(p.numel() for p in lstm_cpu.net.parameters()))
    bundle.lstm = lstm_cpu
    out["ops"] = op_counts(bundle, out["lstm_params"], n_feat)
    del feat
    return out


def main() -> None:
    art = REPO_ROOT / "artifacts"
    tr_dir = art / "traces"
    results = {"literature": [dict(zip(["method", "cite", *DATASETS], row)) for row in LITERATURE_RMSE],
               "costs": vars(Costs()), "thresholds": vars(DecisionThresholds()), "datasets": {}}
    for ds in DATASETS:
        if not (tr_dir / f"{ds}_cv.pkl").exists():
            print(f"[{ds}] no traces, skipping")
            continue
        t0 = time.time()
        meta = json.loads((tr_dir / f"{ds}_meta.json").read_text())
        cv = pd.read_pickle(tr_dir / f"{ds}_cv.pkl")
        ages = {int(k): v for k, v in meta["ages_by_fold"].items()}
        results["datasets"][ds] = {
            "benchmark": benchmark(tr_dir, ds),
            "cv_accuracy": cv_accuracy(cv),
            "decisions": decisions(cv, ages),
            "latency": latency(art, ds),
            "n_cv_units": int(cv["unit"].nunique()),
            "n_cv_cycles": len(cv),
            "ages_by_fold": ages,
            "n_features": meta["n_features"],
            "timings": meta["timings"],
        }
        print(f"[{ds}] evaluated in {time.time() - t0:.0f}s")
    (art / "results").mkdir(exist_ok=True)
    with open(art / "results" / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {art / 'results' / 'results.json'}")


if __name__ == "__main__":
    main()
