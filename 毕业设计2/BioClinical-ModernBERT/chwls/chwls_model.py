"""MVP-3: CH-WLS dual-branch model.

Architecture:
    z (pre-extracted features)
    ├─> mean_head(z)  -> mu       (point predictions)
    └─> var_head(z)   -> log_var  (heteroscedastic log-variance)
                      -> sigma2 = exp(log_var) + eps
                      -> weights w = 1 / sigma2   (for WLS)

The model intentionally does NOT wrap the encoder — it operates entirely
on pre-extracted feature vectors so training is fast and GPU-light.
"""

import torch
import torch.nn as nn


class CHWLSModel(nn.Module):
    """Dual-branch heteroscedastic prediction model.

    Args:
        in_dim:     input feature dimension (e.g. 768 for ModernBERT-base).
        out_dim:    number of outputs (e.g. 14 for Phenotype).
        hidden_dim: width of the hidden layer in both branches.
        dropout:    dropout probability.
        eps:        numerical floor for predicted variance.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps

        self.mean_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.var_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> dict:
        mu = self.mean_head(z)
        log_var = self.var_head(z)
        sigma2 = torch.exp(log_var) + self.eps
        w = 1.0 / sigma2

        return {
            "mu": mu,
            "log_var": log_var,
            "sigma2": sigma2,
            "weights": w,
        }

    @property
    def mean_last_linear(self) -> nn.Linear:
        """Direct access to the final linear layer in mean_head (for WLS replacement)."""
        return self.mean_head[-1]
