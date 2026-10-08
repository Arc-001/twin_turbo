#!/usr/bin/env python3
"""Export the cleaned C-MAPSS data as CSV -- exactly what the models train on.

Cleaning steps (src/data_pipeline/prepare.py, fitted on the training engines):
  1. named columns (unit, cycle, os1-3, s1-s21)
  2. drop sensors that are constant across the training set
  3. assign each row its flight regime (k-means on the 3 settings; 6 regimes in FD002/FD004)
  4. z-score every setting and sensor within its regime
  5. add labels: RUL_true (cycles to failure) and RUL (capped at 125, the training target)

Writes data/clean/{DS}_train.csv and {DS}_test.csv, plus a raw-units copy
({DS}_train_raw.csv) with the same rows and columns before normalization.

Usage:
    python scripts/export_clean_csv.py            # all four datasets
    python scripts/export_clean_csv.py --dataset FD001
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.prepare import load_raw, prepare


def export(ds: str, out: Path) -> None:
    train_raw, test_raw = load_raw(ds, REPO_ROOT / "CMAPSSData")
    data = prepare(ds, train_raw, test_raw)
    cols = ["unit", "cycle", "regime"] + data.feature_cols + ["RUL_true", "RUL"]
    raw_cols = ["unit", "cycle"] + data.feature_cols + ["RUL_true", "RUL"]
    out.mkdir(parents=True, exist_ok=True)
    data.train[cols].to_csv(out / f"{ds}_train.csv", index=False, float_format="%.4f")
    data.test[cols].to_csv(out / f"{ds}_test.csv", index=False, float_format="%.4f")
    train_raw[raw_cols].to_csv(out / f"{ds}_train_raw.csv", index=False)
    dropped = [c for c in [f"s{i}" for i in range(1, 22)] if c not in data.feature_cols]
    print(f"[{ds}] train {len(data.train):>6} rows, test {len(data.test):>6} rows, "
          f"{len(data.feature_cols)} features kept, dropped {dropped}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ALL")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "clean")
    args = p.parse_args()
    for ds in (["FD001", "FD002", "FD003", "FD004"] if args.dataset.upper() == "ALL" else [args.dataset.upper()]):
        export(ds, args.out)


if __name__ == "__main__":
    main()
