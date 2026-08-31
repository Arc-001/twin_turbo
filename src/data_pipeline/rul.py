"""RUL label construction (piecewise-linear degradation assumption)."""

import numpy as np
import pandas as pd


def add_train_rul(df: pd.DataFrame, clip: int | None = 125) -> pd.DataFrame:
    """RUL(t) = last_cycle_of_unit - t, clipped at `clip` (engine assumed healthy above it)."""
    out = df.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    rul = last_cycle - out["cycle"]
    if clip is not None:
        rul = np.minimum(rul, clip)
    out["RUL"] = rul
    return out


def add_test_rul(df: pd.DataFrame, final_rul: np.ndarray, clip: int | None = 125) -> pd.DataFrame:
    """Reconstruct the full RUL trajectory for test units from the known final-cycle RUL.

    `final_rul[i]` is the true RUL at the last recorded cycle of test unit (i + 1),
    ordered as in RUL_FDxxx.txt.
    """
    out = df.copy()
    unit_ids = sorted(out["unit"].unique())
    if len(unit_ids) != len(final_rul):
        raise ValueError(
            f"Got {len(final_rul)} RUL targets but {len(unit_ids)} test units"
        )
    final_rul_by_unit = dict(zip(unit_ids, final_rul))

    last_cycle = out.groupby("unit")["cycle"].transform("max")
    final_rul_col = out["unit"].map(final_rul_by_unit)
    rul = final_rul_col + (last_cycle - out["cycle"])
    if clip is not None:
        rul = np.minimum(rul, clip)
    out["RUL"] = rul
    return out
