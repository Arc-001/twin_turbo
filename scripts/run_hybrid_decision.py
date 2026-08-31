#!/usr/bin/env python3
"""Phase-4: run the hybrid edge/twin decision loop over test-set trajectories.

Simulates cycle-by-cycle streaming inference per engine: edge model runs every
cycle, twin only runs when escalated (low RUL / sharp drop / periodic resync).
Reports twin trigger rate (compute savings) and decision safety (did GROUND_NOW
fire while true RUL was still > 0, i.e. before actual failure).

Usage:
    python scripts/run_hybrid_decision.py --dataset FD001
    python scripts/run_hybrid_decision.py --dataset FD001 --unit 24 --dump-timeline
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_RUL_CLIP, DEFAULT_WINDOW_SIZE, get_variant
from src.data_pipeline.parser import load_run_log, load_rul_targets
from src.data_pipeline.rul import add_test_rul
from src.decision.hybrid import EscalationThresholds, HybridRULEstimator
from src.decision.policy import DecisionThresholds, GROUND_NOW
from src.models.twin_model import TwinRULModel


def simulate_unit(estimator: HybridRULEstimator, seq: np.ndarray, labels: np.ndarray, cycles: np.ndarray,
                   window_size: int) -> list[dict]:
    n = seq.shape[0]
    if n < window_size:
        return []
    timeline = []
    for end in range(window_size - 1, n):
        window = seq[end - window_size + 1: end + 1]
        result = estimator.step(window)
        result["cycle"] = int(cycles[end])
        result["true_rul"] = float(labels[end])
        timeline.append(result)
    return timeline


def summarize_unit(unit_id: int, timeline: list[dict]) -> dict:
    n_cycles = len(timeline)
    n_triggered = sum(1 for r in timeline if r["twin_triggered"])
    ground_now_events = [r for r in timeline if r["zone"] == GROUND_NOW]

    if ground_now_events:
        first = ground_now_events[0]
        # "safe" catch: policy called it before the engine actually ran out of RUL
        safety = "in_time" if first["true_rul"] > 0 else "late"
        ground_now_cycle = first["cycle"]
        ground_now_true_rul = first["true_rul"]
    else:
        safety = "never_triggered"
        ground_now_cycle = None
        ground_now_true_rul = None

    return {
        "unit": unit_id,
        "n_cycles": n_cycles,
        "twin_trigger_rate": n_triggered / n_cycles if n_cycles else 0.0,
        "ground_now_cycle": ground_now_cycle,
        "ground_now_true_rul": ground_now_true_rul,
        "final_true_rul": timeline[-1]["true_rul"] if timeline else None,
        "safety": safety,
    }


def run(dataset: str, unit_filter: int | None, dump_timeline: bool, data_dir: Path,
        processed_dir: Path, artifacts_dir: Path, out_dir: Path) -> None:
    variant = get_variant(dataset)
    window_size = DEFAULT_WINDOW_SIZE

    with open(processed_dir / f"{variant.name}_regime_model.pkl", "rb") as f:
        regime_model = pickle.load(f)
    with open(artifacts_dir / f"{variant.name}_edge_model.pkl", "rb") as f:
        edge_model = pickle.load(f)
    twin_model = TwinRULModel.load(str(artifacts_dir / f"{variant.name}_twin_model.pt"))

    test_df = load_run_log(data_dir / f"test_{variant.name}.txt")
    final_rul = load_rul_targets(data_dir / f"RUL_{variant.name}.txt")
    test_df = add_test_rul(test_df, final_rul, clip=DEFAULT_RUL_CLIP)
    test_norm = regime_model.normalize(test_df)
    feature_cols = regime_model.feature_cols

    units = sorted(test_norm["unit"].unique())
    if unit_filter is not None:
        units = [u for u in units if u == unit_filter]
        if not units:
            raise SystemExit(f"unit {unit_filter} not found in {variant.name} test set")

    escalation = EscalationThresholds()
    thresholds = DecisionThresholds()

    per_unit_summaries = []
    demo_timeline = None
    for unit_id in units:
        unit_df = test_norm[test_norm["unit"] == unit_id].sort_values("cycle")
        seq = unit_df[feature_cols].to_numpy(dtype=float)
        labels = unit_df["RUL"].to_numpy(dtype=float)
        cycles = unit_df["cycle"].to_numpy(dtype=int)

        estimator = HybridRULEstimator(edge_model, twin_model, escalation, thresholds)
        timeline = simulate_unit(estimator, seq, labels, cycles, window_size)
        if not timeline:
            continue
        per_unit_summaries.append(summarize_unit(int(unit_id), timeline))
        if unit_filter is not None:
            demo_timeline = timeline

    n_units = len(per_unit_summaries)
    trigger_rates = [s["twin_trigger_rate"] for s in per_unit_summaries]
    safety_counts = {"in_time": 0, "late": 0, "never_triggered": 0}
    for s in per_unit_summaries:
        safety_counts[s["safety"]] += 1

    aggregate = {
        "dataset": variant.name,
        "n_units": n_units,
        "mean_twin_trigger_rate": float(np.mean(trigger_rates)) if trigger_rates else 0.0,
        "compute_saved_vs_always_twin": 1.0 - (float(np.mean(trigger_rates)) if trigger_rates else 0.0),
        "safety_counts": safety_counts,
        "safety_in_time_pct": safety_counts["in_time"] / n_units * 100 if n_units else 0.0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{variant.name}_hybrid_summary.json", "w") as f:
        json.dump({"aggregate": aggregate, "per_unit": per_unit_summaries}, f, indent=2)

    print(f"[{variant.name}] simulated {n_units} test units")
    print(f"  mean twin trigger rate: {aggregate['mean_twin_trigger_rate']:.1%}"
          f"  (compute saved vs always-twin: {aggregate['compute_saved_vs_always_twin']:.1%})")
    print(f"  GROUND_NOW safety: in_time={safety_counts['in_time']} late={safety_counts['late']}"
          f" never_triggered={safety_counts['never_triggered']}"
          f"  ({aggregate['safety_in_time_pct']:.1f}% in-time)")

    if dump_timeline and demo_timeline is not None:
        path = out_dir / f"{variant.name}_unit{unit_filter}_timeline.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(demo_timeline[0].keys()))
            writer.writeheader()
            for row in demo_timeline:
                row = dict(row)
                row["trigger_reasons"] = ";".join(row["trigger_reasons"])
                writer.writerow(row)
        print(f"  timeline dumped -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid edge/twin decision simulation.")
    parser.add_argument("--dataset", default="FD001", help="FD001 | FD002 | FD003 | FD004 | ALL")
    parser.add_argument("--unit", type=int, default=None, help="Restrict to one test unit")
    parser.add_argument("--dump-timeline", action="store_true", help="Write per-cycle CSV (requires --unit)")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "CMAPSSData")
    parser.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()

    if args.dump_timeline and args.unit is None:
        raise SystemExit("--dump-timeline requires --unit")

    datasets = ["FD001", "FD002", "FD003", "FD004"] if args.dataset.upper() == "ALL" else [args.dataset]
    for name in datasets:
        run(name, args.unit, args.dump_timeline, args.data_dir, args.processed_dir, args.artifacts_dir, args.out_dir)


if __name__ == "__main__":
    main()
