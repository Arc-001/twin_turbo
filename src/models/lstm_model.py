"""Deep sequence baseline: LSTM over the raw sensor window.

Represents the data-driven deep-learning family of RUL models (Zheng et al.,
2017). Consumes the full raw (window_size, n_features) trajectory and reports
uncertainty via MC-dropout (Gal & Ghahramani, 2016). In this project it is a
benchmark and an ablation alternative to the particle-filter digital twin.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class LSTMNet(nn.Module):
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
class LSTMRULModel:
    n_features: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    target_scale: float = 125.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        self.net = LSTMNet(self.n_features, self.hidden_size, self.num_layers, self.dropout).to(self.device)

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
        rng = np.random.RandomState(self.seed)
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
        gen = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
        X_val, y_val = X_t[val_idx].to(self.device), y_scaled[val_idx].to(self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        loss_fn = nn.MSELoss()

        history = []
        best_rmse, best_state = float("inf"), None
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
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            history.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_rmse": val_rmse})
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs}  train_mse={train_loss:.2f}  val_rmse={val_rmse:.2f}")
        if best_state is not None:
            self.net.load_state_dict(best_state)  # keep the best validation epoch
        return history

    def _forward_batched(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=self.device)
                out.append(self.net(xb).cpu().numpy())
        return np.concatenate(out) * self.target_scale

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        return self._forward_batched(X)

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """MC-dropout: keep dropout active across stochastic forward passes."""
        self.net.train()  # dropout stays active
        preds = np.stack([self._forward_batched(X) for _ in range(n_samples)], axis=0)
        self.net.eval()
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
    def load(cls, path: str, device: str | None = None) -> "LSTMRULModel":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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

