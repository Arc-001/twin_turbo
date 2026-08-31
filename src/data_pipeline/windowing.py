"""Sliding-window sequence construction for per-unit sensor trajectories."""

import numpy as np
import pandas as pd


def _padded_sequence(seq: np.ndarray, window_size: int) -> np.ndarray:
    """Left-pad a short trajectory by repeating its first row (standard C-MAPSS practice)."""
    n = seq.shape[0]
    if n >= window_size:
        return seq
    pad = np.repeat(seq[:1], window_size - n, axis=0)
    return np.concatenate([pad, seq], axis=0)


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    label_col: str = "RUL",
    unit_col: str = "unit",
    cycle_col: str = "cycle",
    last_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, unit_ids) from per-cycle rows.

    X: (n_samples, window_size, n_features)
    y: (n_samples,) label at the last cycle of each window
    unit_ids: (n_samples,) originating unit for each window

    last_only=True keeps just the final window per unit (test-set inference mode).
    """
    X_chunks: list[np.ndarray] = []
    y_chunks: list[float] = []
    unit_chunks: list[int] = []

    for unit_id, group in df.sort_values(cycle_col).groupby(unit_col):
        seq = group[feature_cols].to_numpy(dtype=float)
        labels = group[label_col].to_numpy(dtype=float)
        seq = _padded_sequence(seq, window_size)

        if last_only:
            window = seq[-window_size:]
            X_chunks.append(window)
            y_chunks.append(labels[-1])
            unit_chunks.append(unit_id)
            continue

        pad_len = window_size - group.shape[0] if group.shape[0] < window_size else 0
        n_cycles = seq.shape[0]
        for end in range(window_size - 1, n_cycles):
            window = seq[end - window_size + 1 : end + 1]
            label_idx = end - pad_len
            X_chunks.append(window)
            y_chunks.append(labels[max(label_idx, 0)])
            unit_chunks.append(unit_id)

    X = np.stack(X_chunks, axis=0)
    y = np.array(y_chunks, dtype=float)
    unit_ids = np.array(unit_chunks, dtype=int)
    return X, y, unit_ids
