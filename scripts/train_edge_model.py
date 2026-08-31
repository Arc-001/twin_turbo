#!/usr/bin/env python3
"""Phase-2: train + evaluate the edge RUL model on windowed C-MAPSS data.

Usage:
    python scripts/train_edge_model.py --dataset FD001
    python scripts/train_edge_model.py --dataset ALL
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import summarize
from src.models.edge_model import EdgeRULModel
from src.models.features import summarize_windows


def benchmark_latency(model: EdgeRULModel, X: np.ndarray, n_reps: int = 200) -> dict[str, float]:
    """Single-sample predict() timing, as a stand-in for on-device inference latency."""
    sample = X[:1]
    # warm-up
    for _ in range(5):
        model.predict(sample)

    times = []
    for _ in range(n_reps):
        start = time.perf_counter()
        model.predict(sample)
        times.append(time.perf_counter() - start)
    times_ms = np.array(times) * 1000
    return {
        "latency_ms_mean": float(times_ms.mean()),
        "latency_ms_p95": float(np.percentile(times_ms, 95)),
    }


def run(dataset: str, data_dir: Path, out_dir: Path) -> dict:
    data = np.load(data_dir / f"{dataset}.npz", allow_pickle=True)
    X_train_raw, y_train = data["X_train"], data["y_train"]
    X_test_raw, y_test = data["X_test"], data["y_test"]

    X_train = summarize_windows(X_train_raw)
    X_test = summarize_windows(X_test_raw)

    model = EdgeRULModel()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = summarize(y_test, y_pred)
    metrics["latency"] = benchmark_latency(model, X_test)
    metrics["n_train_windows"] = int(X_train.shape[0])
    metrics["n_features"] = int(X_train.shape[1])

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{dataset}_edge_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(out_dir / f"{dataset}_edge_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{dataset}] RMSE={metrics['rmse']:.2f}  PHM08={metrics['phm08_score']:.1f}"
          f"  (per-unit={metrics['phm08_score_per_unit']:.2f})"
          f"  MAE={metrics['mae']:.2f}"
          f"  latency={metrics['latency']['latency_ms_mean']:.3f}ms"
          f" (p95={metrics['latency']['latency_ms_p95']:.3f}ms)")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate edge RUL model.")
    parser.add_argument("--dataset", default="FD001", help="FD001 | FD002 | FD003 | FD004 | ALL")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()

    datasets = ["FD001", "FD002", "FD003", "FD004"] if args.dataset.upper() == "ALL" else [args.dataset]
    for name in datasets:
        run(name, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
