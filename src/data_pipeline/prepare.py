"""One-call data preparation shared by every experiment.

All fitted state (constant-sensor pruning, regime clustering, normalization
stats) is learned from the *training units passed in*, so cross-validation
folds never see statistics from their held-out engines.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DEFAULT_RUL_CLIP, OP_SETTING_COLS, SENSOR_COLS, get_variant
from src.data_pipeline.parser import load_run_log, load_rul_targets
from src.data_pipeline.preprocessing import RegimeModel, find_constant_sensors, fit_regime_model
from src.data_pipeline.rul import add_test_rul, add_train_rul


@dataclass
class PreparedData:
    dataset: str
    feature_cols: list[str]
    sensor_cols: list[str]
    regime_model: RegimeModel
    train: pd.DataFrame  # normalized; RUL (clipped) + RUL_true (unclipped)
    test: pd.DataFrame | None


def load_raw(dataset: str, data_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant = get_variant(dataset)
    train = load_run_log(data_dir / f"train_{variant.name}.txt")
    test = load_run_log(data_dir / f"test_{variant.name}.txt")
    final_rul = load_rul_targets(data_dir / f"RUL_{variant.name}.txt")
    train = add_train_rul(train, clip=None).rename(columns={"RUL": "RUL_true"})
    test = add_test_rul(test, final_rul, clip=None).rename(columns={"RUL": "RUL_true"})
    for df in (train, test):
        df["RUL"] = np.minimum(df["RUL_true"], DEFAULT_RUL_CLIP)
    return train, test


def prepare(
    dataset: str,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame | None = None,
    regime_norm: bool = True,
) -> PreparedData:
    variant = get_variant(dataset)
    constant = find_constant_sensors(train_raw)
    sensor_cols = [s for s in SENSOR_COLS if s not in constant]
    feature_cols = OP_SETTING_COLS + sensor_cols
    n_regimes = variant.n_regimes if regime_norm else 1
    regime_model = fit_regime_model(train_raw, n_regimes, feature_cols)
    return PreparedData(
        dataset=variant.name,
        feature_cols=feature_cols,
        sensor_cols=sensor_cols,
        regime_model=regime_model,
        train=regime_model.normalize(train_raw),
        test=None if test_raw is None else regime_model.normalize(test_raw),
    )
