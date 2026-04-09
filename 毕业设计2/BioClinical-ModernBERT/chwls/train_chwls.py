"""MVP-3: Train the CH-WLS dual-branch model on pre-extracted features.

Two-stage pipeline:
  Stage A — Joint training of mean_head + var_head with Beta-NLL.
  Stage B — (Optional) WLS refinement: freeze var_head, use learned weights
            to solve a global WLS problem, and replace mean_head's last
            linear layer with the WLS-optimal weights.

Usage:
    python -m chwls.train_chwls \
        --features_dir features/phenotype \
        --lr 1e-3 --epochs 100 \
        --beta_nll_beta 0.5 \
        --wls_lambda 1.0 \
        --seed 42
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chwls.chwls_model import CHWLSModel
from chwls.losses import BetaNLL
from chwls.solvers import WLSSolver
from chwls.evaluate import evaluate_all, print_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(features_dir: str, device: torch.device):
    splits = {}
    for name in ("train", "val", "test"):
        data = torch.load(
            os.path.join(features_dir, f"{name}.pt"),
            map_location=device,
            weights_only=True,
        )
        splits[name] = (data["X"], data["y"])
    return splits


# ── Stage A: Joint beta-NLL training ────────────────────────────────────────


def train_stage_a(model, splits, criterion, lr, epochs, batch_size, device):
    """Train mean_head + var_head jointly with Beta-NLL."""
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_f1, best_state = 0.0, None
    patience, patience_counter = 15, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            out = model(xb)
            loss = criterion(out["mu"], out["log_var"], yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(X_train)

        # ── validation ──
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_mu = val_out["mu"].cpu().numpy()
            val_sigma2 = val_out["sigma2"].cpu()

        val_metrics = evaluate_all(val_mu, y_val.cpu().numpy())

        if epoch % 10 == 0 or epoch == 1:
            mean_sigma = val_sigma2.mean().item()
            logger.info(
                "epoch %3d  loss=%.4f  val_wf1=%.4f  val_mse=%.4f  "
                "mean_sigma2=%.4f",
                epoch, avg_loss, val_metrics["weighted_f1"],
                val_metrics["mse"], mean_sigma,
            )

        if val_metrics["weighted_f1"] > best_f1:
            best_f1 = val_metrics["weighted_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ── Stage B: WLS refinement ─────────────────────────────────────────────────


@torch.no_grad()
def refine_with_wls(model, splits, wls_lambda, device):
    """Replace mean_head's last linear layer with WLS-optimal weights."""
    X_train, y_train = splits["train"]

    model.eval()
    out = model(X_train)
    w = out["weights"]  # (N, k)

    # Extract the *input* to the last linear layer of mean_head.
    # mean_head = [Linear, ReLU, Dropout, Linear]
    # We need the hidden representation after ReLU+Dropout (index 0-2).
    feature_extractor = nn.Sequential(*list(model.mean_head.children())[:-1])
    h_train = feature_extractor(X_train)  # (N, hidden_dim)

    solver = WLSSolver(lam=wls_lambda)
    beta = solver(h_train, y_train, w)  # (hidden_dim, k)

    # Overwrite the last linear layer
    last_linear = model.mean_last_linear
    last_linear.weight.copy_(beta.T)
    last_linear.bias.zero_()

    logger.info(
        "WLS refinement complete — beta shape %s, lambda=%.4f",
        list(beta.shape), wls_lambda,
    )
    return model


# ── Evaluation helpers ──────────────────────────────────────────────────────


def save_variance_histogram(sigma2: np.ndarray, out_dir: str, tag: str):
    """Save a histogram of predicted variances to *out_dir*."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sigma2.ravel(), bins=80, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Predicted variance (sigma^2)")
    ax.set_ylabel("Count")
    ax.set_title(f"Variance distribution — {tag}")
    fig.tight_layout()

    safe_tag = tag.replace(" ", "_").replace("/", "-")
    path = os.path.join(out_dir, f"variance_hist_{safe_tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Variance histogram saved -> %s", path)


@torch.no_grad()
def full_evaluate(model, X, y, tag="", out_dir="outputs/chwls"):
    model.eval()
    out = model(X)
    mu = out["mu"].cpu().numpy()
    sigma2 = out["sigma2"].cpu().numpy()
    labels = y.cpu().numpy()

    metrics = evaluate_all(mu, labels)
    metrics["mean_sigma2"] = float(sigma2.mean())
    metrics["median_sigma2"] = float(np.median(sigma2))
    print_metrics(metrics, header=f"{tag}")

    os.makedirs(out_dir, exist_ok=True)
    save_variance_histogram(sigma2, out_dir, tag)

    return metrics


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="MVP-3: Train CH-WLS model")
    parser.add_argument("--features_dir", type=str, default="features/phenotype")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta_nll_beta", type=float, default=0.5)
    parser.add_argument("--wls_lambda", type=float, default=1.0)
    parser.add_argument("--skip_wls_refine", action="store_true",
                        help="Skip Stage B (WLS refinement)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    splits = load_features(args.features_dir, device)
    in_dim = splits["train"][0].shape[1]
    out_dim = splits["train"][1].shape[1]
    logger.info("Feature dim: %d, Output dim: %d", in_dim, out_dim)

    model = CHWLSModel(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = BetaNLL(beta=args.beta_nll_beta)

    # ── Stage A ──
    logger.info("=" * 60)
    logger.info("Stage A: Joint beta-NLL training")
    logger.info("=" * 60)
    model = train_stage_a(
        model, splits, criterion,
        lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, device=device,
    )
    out_dir = os.path.join("outputs", "chwls")

    X_test, y_test = splits["test"]
    full_evaluate(model, X_test, y_test, tag="Stage A — Test (beta-NLL only)", out_dir=out_dir)

    # ── Stage B ──
    if not args.skip_wls_refine:
        logger.info("=" * 60)
        logger.info("Stage B: WLS refinement")
        logger.info("=" * 60)
        model = refine_with_wls(model, splits, args.wls_lambda, device)
        full_evaluate(model, X_test, y_test, tag="Stage B — Test (after WLS refine)", out_dir=out_dir)

    # ── Save ──
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "chwls_model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info("Model saved -> %s", save_path)


if __name__ == "__main__":
    main()
