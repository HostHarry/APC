"""MVP-2: Baseline prediction heads that operate on pre-extracted features."""

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Simple linear projection: (B, in_dim) -> (B, out_dim)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPHead(nn.Module):
    """Two-layer MLP with ReLU and dropout."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RidgeHead:
    """Closed-form Ridge regression (no gradient, no nn.Module).

    Solves:  beta = argmin_b ||Xb - y||^2 + lam * ||b||^2
    via the normal equation  (X^T X + lam I) beta = X^T y.
    """

    def __init__(self):
        self.beta = None

    def fit(self, X: torch.Tensor, y: torch.Tensor, lam: float = 1.0):
        d = X.shape[1]
        A = X.T @ X + lam * torch.eye(d, device=X.device, dtype=X.dtype)
        self.beta = torch.linalg.solve(A, X.T @ y)  # (d, out_dim)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        assert self.beta is not None, "Call fit() before predict()"
        return X @ self.beta

    def search_lambda(self, X_train, y_train, X_val, y_val, candidates=None):
        """Select lambda by minimising MSE on the validation set."""
        if candidates is None:
            candidates = [0.01, 0.1, 1.0, 10.0, 100.0]

        best_lam, best_mse = None, float("inf")
        for lam in candidates:
            self.fit(X_train, y_train, lam=lam)
            preds = self.predict(X_val)
            mse = ((preds - y_val) ** 2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_lam = lam

        self.fit(X_train, y_train, lam=best_lam)
        return best_lam, best_mse
