"""Train every model in the framework on a given set of training engines.

A ModelBundle is what one fleet operator would deploy: the edge regressor with
its conformal calibration, the digital-twin prior + health-index observer, and
the deep LSTM baseline. It is trained twice per dataset in the experiments --
once on all training engines (official-benchmark protocol) and once per CV
fold (run-to-failure decision protocol).
"""

import time
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold

from src.config import DEFAULT_WINDOW_SIZE
from src.data_pipeline.prepare import PreparedData
from src.data_pipeline.windowing import build_windows
from src.models.edge_model import EdgeRULModel
from src.models.features import summarize_windows
from src.models.lstm_model import LSTMRULModel
from src.twin.degradation import FleetPrior, fit_unit
from src.twin.health_index import HealthIndexModel
from src.uncertainty.conformal import MondrianConformal


@dataclass
class ModelBundle:
    data: PreparedData
    edge: EdgeRULModel
    edge_conformal: MondrianConformal
    hi_model: HealthIndexModel
    prior: FleetPrior
    lstm: LSTMRULModel | None
    train_seconds: dict[str, float]


def fit_edge_with_conformal(X_feat, y, groups, alpha: float = 0.1, n_folds: int = 5):
    """Out-of-fold residuals (grouped by engine) calibrate the conformal band,
    then the deployed edge model is refit on everything (CV+ style)."""
    oof = np.zeros_like(y)
    for tr, va in GroupKFold(n_splits=n_folds).split(X_feat, y, groups):
        oof[va] = EdgeRULModel().fit(X_feat[tr], y[tr]).predict(X_feat[va])
    conformal = MondrianConformal(alpha=alpha).fit(oof, y)
    edge = EdgeRULModel().fit(X_feat, y)
    return edge, conformal, oof


def fit_twin_prior(data: PreparedData) -> tuple[HealthIndexModel, FleetPrior]:
    hi_model = HealthIndexModel(data.sensor_cols).fit(data.train)
    fits = []
    for _, g in data.train.groupby("unit"):
        g = g.sort_values("cycle")
        f = fit_unit(g["cycle"].to_numpy(float), hi_model.transform(g))
        if f is not None:
            fits.append(f)
    return hi_model, FleetPrior.from_fits(fits)


def train_bundle(data: PreparedData, lstm_epochs: int = 40, with_lstm: bool = True, seed: int = 0,
                 verbose: bool = False) -> ModelBundle:
    np.random.seed(seed)
    X, y, units = build_windows(data.train, data.feature_cols, DEFAULT_WINDOW_SIZE)
    X_feat = summarize_windows(X)
    timings = {}

    t0 = time.time()
    edge, conformal, _ = fit_edge_with_conformal(X_feat, y, units)
    timings["edge"] = time.time() - t0

    t0 = time.time()
    hi_model, prior = fit_twin_prior(data)
    timings["twin"] = time.time() - t0

    lstm = None
    if with_lstm:
        t0 = time.time()
        lstm = LSTMRULModel(n_features=X.shape[2], seed=seed)
        lstm.fit(X, y, groups=units, epochs=lstm_epochs, verbose=verbose)
        timings["lstm"] = time.time() - t0

    return ModelBundle(data, edge, conformal, hi_model, prior, lstm, timings)
