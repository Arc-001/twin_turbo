"""RUL prediction metrics, including the PHM08 asymmetric scoring function."""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official PHM08 / C-MAPSS scoring fn: penalizes late predictions harder than early.

    d = predicted - true.
    d < 0 (early / conservative):  exp(-d/13) - 1
    d >= 0 (late / dangerous):     exp( d/10) - 1
    Summed over all samples (not averaged) -- matches the original challenge definition.
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "phm08_score": phm08_score(y_true, y_pred),
        "phm08_score_per_unit": phm08_score(y_true, y_pred) / len(y_true),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }
