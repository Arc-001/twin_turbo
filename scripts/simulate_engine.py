#!/usr/bin/env python3
"""Stream one engine through the live hybrid estimator, cycle by cycle.

This is the deployable path (HybridRULEstimator): edge inference every cycle,
buffered health-index observations, and a particle-filter twin sync only when
a trigger fires. Useful for inspecting a single engine and for exporting demo
timelines.

Usage:
    python scripts/simulate_engine.py --dataset FD001 --unit 34
    python scripts/simulate_engine.py --dataset FD004 --unit 12 --split test --csv out.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.prepare import load_raw
from src.decision.hybrid import HybridRULEstimator
from src.experiments.traces import unit_windows
from src.twin.particle_filter import ParticleTwin


def simulate(bundle, unit_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    data = bundle.data
    est = HybridRULEstimator(bundle.edge, bundle.edge_conformal, bundle.hi_model,
                             ParticleTwin(bundle.prior, seed=seed), feature_cols=data.feature_cols)
    g = unit_df.sort_values("cycle")
    windows = unit_windows(g[data.feature_cols].to_numpy(float))
    rows = []
    for w, cycle, rul_true in zip(windows, g["cycle"], g["RUL_true"]):
        r = est.step(w, int(cycle))
        r["trigger_reasons"] = ";".join(r["trigger_reasons"])
        r["rul_true"] = float(rul_true)
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="FD001")
    p.add_argument("--unit", type=int, required=True)
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    with open(REPO_ROOT / "artifacts" / "models" / f"{args.dataset}_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    train_raw, test_raw = load_raw(args.dataset, REPO_ROOT / "CMAPSSData")
    raw = test_raw if args.split == "test" else train_raw
    unit_df = bundle.data.regime_model.normalize(raw[raw["unit"] == args.unit])
    if unit_df.empty:
        raise SystemExit(f"unit {args.unit} not in {args.dataset} {args.split}")
    tl = simulate(bundle, unit_df, seed=args.unit)

    syncs = tl["twin_synced"].sum()
    print(f"[{args.dataset} unit {args.unit}] {len(tl)} cycles, {syncs} twin syncs ({syncs / len(tl):.0%})")
    print(tl[["cycle", "rul_true", "edge_rul", "final_rul", "final_lo", "source", "zone", "trigger_reasons"]]
          .tail(12).round(1).to_string(index=False))
    if args.csv:
        tl.to_csv(args.csv, index=False)
        print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
