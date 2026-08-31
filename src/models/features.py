"""Window -> compact feature vector, for the tree-based edge model.

Raw (window_size, n_sensors) windows are collapsed to per-sensor summary
stats (last value, mean, std, linear trend) so the edge model stays a small
tabular regressor instead of needing a sequence model on-device.
"""

import numpy as np


def summarize_windows(X: np.ndarray) -> np.ndarray:
    """X: (n_samples, window_size, n_features) -> (n_samples, 4 * n_features)."""
    n_samples, window_size, n_features = X.shape

    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)

    t = np.arange(window_size, dtype=float)
    t_centered = t - t.mean()
    denom = np.sum(t_centered**2)
    x_centered = X - mean[:, None, :]
    slope = np.sum(x_centered * t_centered[None, :, None], axis=1) / denom

    return np.concatenate([last, mean, std, slope], axis=1)


def feature_names(feature_cols: list[str]) -> list[str]:
    return (
        [f"{c}_last" for c in feature_cols]
        + [f"{c}_mean" for c in feature_cols]
        + [f"{c}_std" for c in feature_cols]
        + [f"{c}_slope" for c in feature_cols]
    )
