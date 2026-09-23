"""Per-cycle traces: every model's estimate at every cycle of every engine.

The twin's posterior at cycle t does not depend on *when* it was synchronized
(lazy assimilation ingests the same observations), so a trace computed with
the twin synced every cycle can be sub-sampled to replay any escalation
schedule exactly. That makes every decision policy and ablation a cheap
replay over the same cached trace.
"""

import numpy as np
import pandas as pd

from src.config import DEFAULT_RUL_CLIP, DEFAULT_WINDOW_SIZE
from src.experiments.bundle import ModelBundle
from src.models.features import summarize_windows
from src.twin.particle_filter import ParticleTwin

MC_SAMPLES = 20


def unit_windows(seq: np.ndarray, window_size: int = DEFAULT_WINDOW_SIZE) -> np.ndarray:
    """One window ending at every cycle (left-padded at the start of life)."""
    padded = np.concatenate([np.repeat(seq[:1], window_size - 1, axis=0), seq], axis=0)
    idx = np.arange(seq.shape[0])[:, None] + np.arange(window_size)[None, :]
    return padded[idx]


def _weighted_stats(rul: np.ndarray, w: np.ndarray, clip: float) -> tuple[float, float, float, float, float]:
    order = np.argsort(rul)
    r, cdf = rul[order], np.cumsum(w[order])
    q = [r[min(np.searchsorted(cdf, p), len(r) - 1)] for p in (0.05, 0.5, 0.95)]
    mean_clip = float(np.sum(w * np.minimum(rul, clip)))
    mean = float(np.sum(w * rul))
    std = float(np.sqrt(np.sum(w * (rul - mean) ** 2)))
    return mean_clip, std, q[0], q[1], q[2]


def twin_trace(bundle: ModelBundle, t: np.ndarray, hi: np.ndarray, seed: int) -> np.ndarray:
    twin = ParticleTwin(bundle.prior, seed=seed)
    out = np.zeros((len(t), 5))
    for i in range(len(t)):
        twin.assimilate(t[i : i + 1], hi[i : i + 1])
        out[i] = _weighted_stats(twin.rul_particles(), twin.weights, DEFAULT_RUL_CLIP)
    return out


def build_traces(bundle: ModelBundle, df: pd.DataFrame, last_only: bool = False) -> pd.DataFrame:
    """df: normalized frame with unit, cycle, RUL, RUL_true. Returns one row per
    (unit, cycle) -- or one row per unit at its last cycle if last_only."""
    data = bundle.data
    rows, windows = [], []
    twin_rows = []
    for unit, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        seq = g[data.feature_cols].to_numpy(float)
        w = unit_windows(seq)
        t = g["cycle"].to_numpy(float)
        hi = bundle.hi_model.transform(g)
        tw = twin_trace(bundle, t, hi, seed=int(unit))
        sel = slice(-1, None) if last_only else slice(None)
        windows.append(w[sel])
        twin_rows.append(tw[sel])
        rows.append(pd.DataFrame({
            "unit": int(unit), "cycle": g["cycle"].to_numpy()[sel],
            "rul": g["RUL"].to_numpy(float)[sel], "rul_true": g["RUL_true"].to_numpy(float)[sel],
            "hi": hi[sel],
        }))
    out = pd.concat(rows, ignore_index=True)
    X = np.concatenate(windows)
    twin = np.concatenate(twin_rows)

    out["edge"] = bundle.edge.predict(summarize_windows(X))
    out["edge_lo"], out["edge_hi"] = bundle.edge_conformal.interval(out["edge"].to_numpy())
    if bundle.lstm is not None:
        out["lstm"] = bundle.lstm.predict(X)
        out["lstm_mc_mean"], out["lstm_mc_std"] = bundle.lstm.predict_with_uncertainty(X, MC_SAMPLES)
    out["twin"], out["twin_std"], out["twin_q05"], out["twin_q50"], out["twin_q95"] = twin.T
    return out
