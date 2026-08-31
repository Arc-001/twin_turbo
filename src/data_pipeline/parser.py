"""Raw C-MAPSS text file readers."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ALL_COLS


def load_run_log(path: str | Path) -> pd.DataFrame:
    """Load a train_FDxxx.txt or test_FDxxx.txt file into a labeled DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_rul_targets(path: str | Path) -> np.ndarray:
    """Load a RUL_FDxxx.txt file: one RUL value per test unit, ordered by unit id."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])["RUL"].to_numpy()
