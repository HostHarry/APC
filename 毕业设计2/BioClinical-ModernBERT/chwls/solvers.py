"""MVP-3: Differentiable Ridge / WLS solvers.

All solvers use torch.linalg.solve (never torch.inverse) and monitor
the condition number of the system matrix for early warning of instability.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

COND_WARN_THRESHOLD = 1e10


def _check_cond(A: torch.Tensor, label: str = ""):
    """Log a warning when the condition number is dangerously high."""
    with torch.no_grad():
        cond = torch.linalg.cond(A).item()
    if cond > COND_WARN_THRESHOLD:
        logger.warning(
            "%s condition number %.2e exceeds threshold %.2e — consider increasing lambda",
            label, cond, COND_WARN_THRESHOLD,
        )
    return cond


class RidgeSolver(nn.Module):
    """Global Ridge regression solver (differentiable).

    Given full-dataset X (N, d) and y (N, k), computes:

        beta = (X^T X + lam I)^{-1} X^T y        via linalg.solve

    The solver is differentiable w.r.t. X and y so that it can sit inside
    a training loop that fine-tunes upstream parameters.
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = X.shape[1]
        A = X.T @ X + self.lam * torch.eye(d, device=X.device, dtype=X.dtype)
        _check_cond(A, "RidgeSolver")
        beta = torch.linalg.solve(A, X.T @ y)  # (d, k)
        return beta


class WLSSolver(nn.Module):
    """Differentiable Weighted Least Squares solver.

    For each output column j it solves:

        beta_j = (X^T W_j X + lam I)^{-1} X^T W_j y_j

    where W_j = diag(w[:, j]) are per-sample, per-output weights typically
    derived from the inverse predicted variance.

    For multi-output problems (k > 1) the k systems are solved one at a time
    since each has its own weight vector. The 768x768 matrices are small
    enough that serial solving is negligible.
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            X: (N, d) feature matrix.
            y: (N, k) targets.
            w: (N, k) per-sample-per-output weights (positive).

        Returns:
            beta: (d, k) weight matrix.
        """
        N, d = X.shape
        k = y.shape[1]
        reg = self.lam * torch.eye(d, device=X.device, dtype=X.dtype)

        betas = []
        for j in range(k):
            wj = w[:, j]                          # (N,)
            Xw = X * wj.unsqueeze(1)              # (N, d)  — X diag(w) implicitly
            A = Xw.T @ X + reg                    # (d, d)
            b = Xw.T @ y[:, j]                    # (d,)
            _check_cond(A, f"WLSSolver[output={j}]")
            beta_j = torch.linalg.solve(A, b)     # (d,)
            betas.append(beta_j)

        return torch.stack(betas, dim=1)           # (d, k)
