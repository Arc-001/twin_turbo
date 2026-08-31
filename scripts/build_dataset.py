#!/usr/bin/env python3
"""Phase-1 pipeline entry point: raw C-MAPSS text -> windowed train/test arrays.

Usage:
    python scripts/build_dataset.py --dataset FD001
    python scripts/build_dataset.py --dataset FD004 --window 30 --clip 125
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import DEFAULT_RUL_CLIP, DEFAULT_WINDOW_SIZE, OP_SETTING_COLS, SENSOR_COLS, get_variant
from src.data_pipeline.parser import load_run_log, load_rul_targets
from src.data_pipeline.preprocessing import find_constant_sensors, fit_regime_model
from src.data_pipeline.rul import add_test_rul, add_train_rul
from src.data_pipeline.windowing import build_windows


def build(dataset: str, window_size: int, rul_clip: int, data_dir: Path, out_dir: Path) -> None:
    variant = get_variant(dataset)

    train_df = load_run_log(data_dir / f"train_{variant.name}.txt")
    test_df = load_run_log(data_dir / f"test_{variant.name}.txt")
    test_final_rul = load_rul_targets(data_dir / f"RUL_{variant.name}.txt")

    train_df = add_train_rul(train_df, clip=rul_clip)
    test_df = add_test_rul(test_df, test_final_rul, clip=rul_clip)

    constant_sensors = find_constant_sensors(train_df)
    feature_cols = OP_SETTING_COLS + [s for s in SENSOR_COLS if s not in constant_sensors]

    regime_model = fit_regime_model(train_df, variant.n_regimes, feature_cols)
    train_norm = regime_model.normalize(train_df)
    test_norm = regime_model.normalize(test_df)

    X_train, y_train, unit_train = build_windows(
        train_norm, feature_cols, window_size, last_only=False
    )
    X_test, y_test, unit_test = build_windows(
        test_norm, feature_cols, window_size, last_only=True
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{variant.name}.npz",
        X_train=X_train,
        y_train=y_train,
        unit_train=unit_train,
        X_test=X_test,
        y_test=y_test,
        unit_test=unit_test,
        feature_cols=np.array(feature_cols),
    )
    with open(out_dir / f"{variant.name}_regime_model.pkl", "wb") as f:
        pickle.dump(regime_model, f)

    print(f"[{variant.name}] regimes={variant.n_regimes} faults={variant.n_fault_modes}")
    print(f"  dropped constant sensors: {constant_sensors}")
    print(f"  features used ({len(feature_cols)}): {feature_cols}")
    print(f"  train windows: X={X_train.shape} y={y_train.shape}")
    print(f"  test windows:  X={X_test.shape} y={y_test.shape}")
    print(f"  saved -> {out_dir / f'{variant.name}.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build windowed C-MAPSS train/test arrays.")
    parser.add_argument("--dataset", default="FD001", help="FD001 | FD002 | FD003 | FD004 | ALL")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--clip", type=int, default=DEFAULT_RUL_CLIP)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "CMAPSSData")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    args = parser.parse_args()

    datasets = ["FD001", "FD002", "FD003", "FD004"] if args.dataset.upper() == "ALL" else [args.dataset]
    for name in datasets:
        build(name, args.window, args.clip, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
