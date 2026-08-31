"""Sensor pruning, operating-regime clustering, and regime-aware normalization."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import CONSTANT_SENSOR_STD_THRESHOLD, OP_SETTING_COLS, SENSOR_COLS


def find_constant_sensors(train_df: pd.DataFrame, threshold: float = CONSTANT_SENSOR_STD_THRESHOLD) -> list[str]:
    """Sensors with ~zero variance across the whole training set carry no signal."""
    stds = train_df[SENSOR_COLS].std()
    return stds[stds < threshold].index.tolist()


@dataclass
class RegimeModel:
    """Operating-condition clustering + per-regime feature normalization stats."""

    n_regimes: int
    kmeans: KMeans | None
    feature_cols: list[str]
    regime_mean: dict[int, np.ndarray]
    regime_std: dict[int, np.ndarray]

    def assign_regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_regimes <= 1:
            return np.zeros(len(df), dtype=int)
        assert self.kmeans is not None
        return self.kmeans.predict(df[OP_SETTING_COLS].to_numpy())

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        regimes = self.assign_regimes(df)
        out["regime"] = regimes
        values = out[self.feature_cols].to_numpy(dtype=float)
        for regime_id in np.unique(regimes):
            mask = regimes == regime_id
            mean = self.regime_mean[regime_id]
            std = self.regime_std[regime_id]
            values[mask] = (values[mask] - mean) / std
        out[self.feature_cols] = values
        return out


def fit_regime_model(train_df: pd.DataFrame, n_regimes: int, feature_cols: list[str]) -> RegimeModel:
    """Fit regime clustering (if n_regimes > 1) and per-regime mean/std on the training set."""
    if n_regimes <= 1:
        regimes = np.zeros(len(train_df), dtype=int)
        kmeans = None
    else:
        kmeans = KMeans(n_clusters=n_regimes, n_init=10, random_state=0)
        regimes = kmeans.fit_predict(train_df[OP_SETTING_COLS].to_numpy())

    regime_mean: dict[int, np.ndarray] = {}
    regime_std: dict[int, np.ndarray] = {}
    values = train_df[feature_cols].to_numpy(dtype=float)
    for regime_id in np.unique(regimes):
        mask = regimes == regime_id
        mean = values[mask].mean(axis=0)
        std = values[mask].std(axis=0)
        std[std < 1e-8] = 1.0  # guard against degenerate/constant columns within a regime
        regime_mean[regime_id] = mean
        regime_std[regime_id] = std

    return RegimeModel(
        n_regimes=n_regimes,
        kmeans=kmeans,
        feature_cols=feature_cols,
        regime_mean=regime_mean,
        regime_std=regime_std,
    )
