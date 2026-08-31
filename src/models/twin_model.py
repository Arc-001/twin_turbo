"""Digital twin sequence model: LSTM over the raw sensor window.

Unlike the edge model (flattened summary stats -> tree), the twin consumes
the full raw (window_size, n_features) trajectory to capture temporal
dynamics, and supports:
  - MC-dropout uncertainty (stochastic forward passes with dropout active)
  - forward trajectory projection (iterative what-if simulation to failure)
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class TwinLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.head(self.drop(last_hidden))
        return out.squeeze(-1)


@dataclass
class TwinRULModel:
    n_features: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    target_scale: float = 125.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.net = TwinLSTM(self.n_features, self.hidden_size, self.num_layers, self.dropout).to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        epochs: int = 15,
        batch_size: int = 256,
        val_split: float = 0.1,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        n = X.shape[0]
        rng = np.random.RandomState(0)
        if groups is not None:
            # Split by unit (engine), not by window -- windows from the same
            # unit are highly correlated, so a random per-window split leaks
            # unit identity into validation and overstates generalization.
            unique_groups = rng.permutation(np.unique(groups))
            n_val_groups = max(1, int(len(unique_groups) * val_split))
            val_group_set = set(unique_groups[:n_val_groups].tolist())
            val_mask = np.isin(groups, list(val_group_set))
            val_idx = np.where(val_mask)[0]
            train_idx = np.where(~val_mask)[0]
        else:
            idx = rng.permutation(n)
            n_val = int(n * val_split)
            val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_t = torch.tensor(X, dtype=torch.float32)
        y_scaled = torch.tensor(y, dtype=torch.float32) / self.target_scale
        train_ds = torch.utils.data.TensorDataset(X_t[train_idx], y_scaled[train_idx])
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        X_val, y_val = X_t[val_idx].to(self.device), y_scaled[val_idx].to(self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        loss_fn = nn.MSELoss()

        history = []
        for epoch in range(epochs):
            self.net.train()
            running = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
            train_loss = running / len(train_idx)

            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(X_val)
                val_rmse_scaled = torch.sqrt(loss_fn(val_pred, y_val)).item()
            sched.step(val_rmse_scaled)
            val_rmse = val_rmse_scaled * self.target_scale
            history.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_rmse": val_rmse})
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs}  train_mse={train_loss:.2f}  val_rmse={val_rmse:.2f}")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.net(X_t).cpu().numpy() * self.target_scale

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """MC-dropout: keep dropout active across stochastic forward passes."""
        self.net.train()  # dropout stays active
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.net(X_t).cpu().numpy())
        preds = np.stack(preds, axis=0) * self.target_scale
        return preds.mean(axis=0), preds.std(axis=0)

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "n_features": self.n_features,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "target_scale": self.target_scale,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TwinRULModel":
        ckpt = torch.load(path, map_location=device)
        model = cls(
            n_features=ckpt["n_features"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
            target_scale=ckpt.get("target_scale", 125.0),
            device=device,
        )
        model.net.load_state_dict(ckpt["state_dict"])
        return model


def project_trajectory(
    model: TwinRULModel,
    window: np.ndarray,
    max_horizon: int = 300,
    rul_floor: float = 0.0,
) -> np.ndarray:
    """Digital-twin what-if forward sim: roll the window forward using each
    feature's linear trend within the current window, re-predicting RUL each
    step, until predicted RUL hits `rul_floor` or `max_horizon` cycles pass.

    window: (window_size, n_features) -- most recent real observations.
    Returns: (n_projected_steps,) array of projected RUL values.
    """
    window_size, n_features = window.shape
    t = np.arange(window_size, dtype=float)
    t_centered = t - t.mean()
    denom = np.sum(t_centered**2)

    cur = window.copy()
    projected = []
    for _ in range(max_horizon):
        mean = cur.mean(axis=0)
        slope = np.sum((cur - mean) * t_centered[:, None], axis=0) / denom
        next_step = cur[-1] + slope  # linear extrapolation of each sensor/setting

        cur = np.concatenate([cur[1:], next_step[None, :]], axis=0)
        rul_pred = float(model.predict(cur[None, :, :])[0])
        projected.append(rul_pred)
        if rul_pred <= rul_floor:
            break

    return np.array(projected)
