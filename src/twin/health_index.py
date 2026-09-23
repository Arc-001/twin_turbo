"""Health-index (HI) observer: fuses regime-normalized sensors into one scalar.

Follows the similarity-based prognostics recipe of Wang et al. (PHM08 winner):
a linear map is fitted so that early-life cycles score HI = 1 and the final
cycles before failure score HI = 0. Cycles in between are never used as
targets, so the *shape* of the HI curve (typically exponential) comes from the
sensors themselves, not from a label assumption.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class HealthIndexModel:
    sensor_cols: list[str]
    healthy_cycles: int = 30
    failed_cycles: int = 10

    def fit(self, train_norm: pd.DataFrame) -> "HealthIndexModel":
        rows, targets = [], []
        for _, g in train_norm.groupby("unit"):
            g = g.sort_values("cycle")
            x = g[self.sensor_cols].to_numpy(dtype=float)
            rows.append(x[: self.healthy_cycles])
            targets.append(np.ones(min(self.healthy_cycles, len(x))))
            rows.append(x[-self.failed_cycles:])
            targets.append(np.zeros(min(self.failed_cycles, len(x))))
        self.reg = LinearRegression().fit(np.concatenate(rows), np.concatenate(targets))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(df[self.sensor_cols].to_numpy(dtype=float))

    def transform_array(self, x: np.ndarray) -> np.ndarray:
        """x: (..., n_sensors) already ordered as sensor_cols."""
        return x @ self.reg.coef_ + self.reg.intercept_

    @property
    def weights(self) -> dict[str, float]:
        return dict(zip(self.sensor_cols, self.reg.coef_.tolist()))
