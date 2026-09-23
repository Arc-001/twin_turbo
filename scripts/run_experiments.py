#!/usr/bin/env python3
"""Train every model under both evaluation protocols and cache per-cycle traces.

Protocol A -- official benchmark:
    train on all training engines, predict the last cycle of every official
    test engine. Directly comparable with published C-MAPSS results.
Protocol B -- run-to-failure decision study:
    5-fold cross-validation over *training engines* (grouped by engine). Each
    held-out engine is replayed from its first cycle all the way to failure,
    which the truncated official test set cannot support.

Outputs (artifacts/):
    traces/{DS}_test.pkl      one row per official test engine
    traces/{DS}_cv.pkl        one row per (held-out engine, cycle), with fold id
    traces/{DS}_meta.json     fold-wise age-replacement ages, timings, sizes
    models/{DS}_bundle.pkl    protocol-A bundle (edge, conformal, twin prior, HI observer)
    models/{DS}_lstm.pt       protocol-A LSTM baseline

Usage:
    python scripts/run_experiments.py --dataset ALL
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.prepare import load_raw, prepare
from src.decision.replay import optimal_age
from src.experiments.bundle import train_bundle
from src.experiments.traces import build_traces

N_FOLDS = 5
SEED = 0


def run(dataset: str, epochs: int, n_folds: int, data_dir: Path, out_dir: Path) -> None:
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    train_raw, test_raw = load_raw(dataset, data_dir)
    meta: dict = {"dataset": dataset, "timings": {}}

    # ---- Protocol A: official benchmark ------------------------------------
    t0 = time.time()
    data = prepare(dataset, train_raw, test_raw)
    bundle = train_bundle(data, lstm_epochs=epochs, seed=SEED)
    test_traces = build_traces(bundle, data.test, last_only=True)
    test_traces.to_pickle(out_dir / "traces" / f"{dataset}_test.pkl")
    meta["timings"]["benchmark"] = {**bundle.train_seconds, "total": time.time() - t0}
    bundle.lstm.save(str(out_dir / "models" / f"{dataset}_lstm.pt"))
    lstm, bundle.lstm = bundle.lstm, None
    with open(out_dir / "models" / f"{dataset}_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
    bundle.lstm = lstm
    print(f"[{dataset}] benchmark done in {time.time() - t0:.0f}s")

    # ---- Ablation: global instead of per-regime normalization --------------
    data_g = prepare(dataset, train_raw, test_raw, regime_norm=False)
    bundle_g = train_bundle(data_g, with_lstm=False, seed=SEED)
    build_traces(bundle_g, data_g.test, last_only=True).to_pickle(
        out_dir / "traces" / f"{dataset}_test_globalnorm.pkl")

    # ---- Protocol B: grouped CV, run-to-failure replay ----------------------
    units = np.sort(train_raw["unit"].unique())
    folds = []
    meta["ages_by_fold"] = {}
    for k, (fit_idx, hold_idx) in enumerate(KFold(n_folds, shuffle=True, random_state=SEED).split(units)):
        t0 = time.time()
        fit_units, hold_units = units[fit_idx], units[hold_idx]
        fit_raw = train_raw[train_raw["unit"].isin(fit_units)]
        hold_raw = train_raw[train_raw["unit"].isin(hold_units)]
        data_k = prepare(dataset, fit_raw, hold_raw)
        bundle_k = train_bundle(data_k, lstm_epochs=epochs, seed=SEED + k)
        tr = build_traces(bundle_k, data_k.test)
        tr["fold"] = k
        folds.append(tr)
        lifetimes = fit_raw.groupby("unit")["cycle"].max().to_numpy()
        meta["ages_by_fold"][k] = optimal_age(lifetimes)
        print(f"[{dataset}] fold {k + 1}/{n_folds}: {len(hold_units)} held-out engines, "
              f"{len(tr)} cycles, {time.time() - t0:.0f}s")
    pd.concat(folds, ignore_index=True).to_pickle(out_dir / "traces" / f"{dataset}_cv.pkl")

    meta["n_features"] = len(data.feature_cols)
    meta["feature_cols"] = data.feature_cols
    meta["hi_weights"] = bundle.hi_model.weights
    meta["prior"] = {"mean": bundle.prior.mean.tolist(), "cov": bundle.prior.cov.tolist(),
                     "h_fail_mean": bundle.prior.h_fail_mean, "h_fail_std": bundle.prior.h_fail_std,
                     "obs_std": bundle.prior.obs_std}
    meta["conformal"] = {"bins": [float(b) for b in bundle.edge_conformal.bin_edges],
                         "q_lo": bundle.edge_conformal.q_lo.tolist(), "q_hi": bundle.edge_conformal.q_hi.tolist()}
    with open(out_dir / "traces" / f"{dataset}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ALL", help="FD001 | FD002 | FD003 | FD004 | ALL")
    p.add_argument("--epochs", type=int, default=40, help="LSTM baseline epochs")
    p.add_argument("--folds", type=int, default=N_FOLDS)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "CMAPSSData")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = p.parse_args()
    datasets = ["FD001", "FD002", "FD003", "FD004"] if args.dataset.upper() == "ALL" else [args.dataset.upper()]
    for ds in datasets:
        run(ds, args.epochs, args.folds, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
