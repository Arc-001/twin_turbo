"""Lightweight tabular RUL regressor -- the "edge intelligence" baseline.

Tree-based, trained on compact per-window summary features, sized for fast
single-sample inference (edge-latency proxy in scripts/train_edge_model.py).
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class EdgeRULModel:
    max_iter: int = 150
    max_depth: int = 6
    learning_rate: float = 0.08
    random_state: int = 0

    def __post_init__(self) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EdgeRULModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
