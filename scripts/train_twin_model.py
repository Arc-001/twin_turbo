#!/usr/bin/env python3
"""Phase-3: train + evaluate the digital-twin LSTM, and demo trajectory projection.

Usage:
    python scripts/train_twin_model.py --dataset FD001
    python scripts/train_twin_model.py --dataset ALL --epochs 15
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import summarize
from src.models.twin_model import TwinRULModel, project_trajectory


def run(dataset: str, epochs: int, data_dir: Path, out_dir: Path) -> dict:
    data = np.load(data_dir / f"{dataset}.npz", allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    unit_train = data["unit_train"]
    unit_test = data["unit_test"]

    print(f"[{dataset}] training twin LSTM  X_train={X_train.shape}")
    model = TwinRULModel(n_features=X_train.shape[2])
    start = time.time()
    history = model.fit(X_train, y_train, groups=unit_train, epochs=epochs)
    train_seconds = time.time() - start

    y_pred = model.predict(X_test)
    metrics = summarize(y_test, y_pred)
    metrics["train_seconds"] = train_seconds
    metrics["n_train_windows"] = int(X_train.shape[0])

    # MC-dropout uncertainty on the test set -- feeds the phase-4 edge->twin trigger.
    _, y_std = model.predict_with_uncertainty(X_test, n_samples=20)
    metrics["uncertainty"] = {
        "mean_std": float(y_std.mean()),
        "max_std": float(y_std.max()),
    }

    # Forward what-if projection demo: take the first test unit's final window,
    # project forward to see how far the twin thinks it is from failure.
    demo_window = X_test[0]
    trajectory = project_trajectory(model, demo_window)
    metrics["trajectory_demo"] = {
        "unit": int(unit_test[0]),
        "true_rul": float(y_test[0]),
        "projected_cycles_to_floor": int(len(trajectory)),
        "projected_rul_curve_head": [round(v, 2) for v in trajectory[:10].tolist()],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / f"{dataset}_twin_model.pt"))
    with open(out_dir / f"{dataset}_twin_metrics.json", "w") as f:
        json.dump({"history": history, **metrics}, f, indent=2)

    print(f"[{dataset}] RMSE={metrics['rmse']:.2f}  PHM08={metrics['phm08_score']:.1f}"
          f"  (per-unit={metrics['phm08_score_per_unit']:.2f})"
          f"  MAE={metrics['mae']:.2f}"
          f"  train_time={train_seconds:.1f}s"
          f"  mc_dropout_std(mean)={metrics['uncertainty']['mean_std']:.2f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate digital twin LSTM.")
    parser.add_argument("--dataset", default="FD001", help="FD001 | FD002 | FD003 | FD004 | ALL")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()

    datasets = ["FD001", "FD002", "FD003", "FD004"] if args.dataset.upper() == "ALL" else [args.dataset]
    for name in datasets:
        run(name, args.epochs, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
