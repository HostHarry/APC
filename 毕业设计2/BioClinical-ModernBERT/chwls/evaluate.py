"""Shared evaluation utilities for MVP-2 and MVP-3.

Returns both regression metrics (MSE, MAE) and multi-label classification
metrics (weighted F1, micro F1, macro AUROC) so we can track both views.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)


def evaluate_all(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate predictions against ground-truth labels.

    Args:
        preds:  (N, K) float array — raw logits / scores (before sigmoid).
        labels: (N, K) float array — ground-truth multi-hot (0/1).

    Returns:
        dict with regression and classification metrics.
    """
    results = {}

    # --- Regression view ---
    results["mse"] = mean_squared_error(labels, preds)
    results["mae"] = mean_absolute_error(labels, preds)

    # --- Classification view ---
    binary_preds = (preds > 0).astype(int)
    binary_labels = labels.astype(int)

    results["weighted_f1"] = f1_score(
        binary_labels, binary_preds, average="weighted", zero_division=0
    )
    results["micro_f1"] = f1_score(
        binary_labels, binary_preds, average="micro", zero_division=0
    )

    # macro AUROC — skip labels that are all-zero or all-one in ground truth
    try:
        results["macro_auroc"] = roc_auc_score(
            binary_labels, preds, average="macro", multi_class="ovr"
        )
    except ValueError:
        results["macro_auroc"] = float("nan")

    return results


def print_metrics(metrics: dict, header: str = ""):
    if header:
        print(f"\n{'='*60}")
        print(f"  {header}")
        print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.6f}")
    print()
