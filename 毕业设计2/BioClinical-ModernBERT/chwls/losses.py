"""MVP-3: Heteroscedastic loss functions for CH-WLS training.

GaussianNLL  — standard heteroscedastic Gaussian negative log-likelihood.
BetaNLL      — gradient-decoupled variant that prevents the variance branch
               from dragging the mean branch off course.

Reference (beta-NLL):
    Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty Estimation
    with Probabilistic Neural Networks", ICLR 2022.
"""

import torch
import torch.nn as nn


class GaussianNLL(nn.Module):
    """Heteroscedastic Gaussian NLL.

    L = 0.5 * [ log(sigma^2) + (y - mu)^2 / sigma^2 ]

    Args:
        eps: small constant added to variance for numerical stability.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var) + self.eps
        nll = 0.5 * (log_var + (y - mu) ** 2 / var)
        return nll.mean()


class BetaNLL(nn.Module):
    """Beta-NLL: stabilised heteroscedastic loss.

    Detaches variance before using it to re-weight the NLL, which decouples
    the mean and variance gradients and prevents the common failure mode
    where the variance branch collapses or diverges.

    L_i = [ 0.5 * (log var_i + (y_i - mu_i)^2 / var_i) ] / var_i.detach()^beta

    Args:
        beta:  re-weighting exponent (default 0.5, from the original paper).
        eps:   numerical stability constant.
    """

    def __init__(self, beta: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var) + self.eps
        nll = 0.5 * (log_var + (y - mu) ** 2 / var)
        weights = var.detach() ** self.beta
        return (nll / weights).mean()
