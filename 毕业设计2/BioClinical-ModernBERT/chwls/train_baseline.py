"""MVP-2: Train and evaluate baseline heads on pre-extracted features.

Usage:
    python -m chwls.train_baseline \
        --features_dir features/phenotype \
        --head mlp \
        --lr 1e-3 --epochs 50 --seed 42
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from chwls.heads import LinearHead, MLPHead, RidgeHead
from chwls.evaluate import evaluate_all, print_metrics


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(features_dir: str, device: torch.device):
    splits = {}
    for name in ("train", "val", "test"):
        data = torch.load(os.path.join(features_dir, f"{name}.pt"), map_location=device, weights_only=True)
        splits[name] = (data["X"], data["y"])
    return splits


def train_neural_head(head, splits, lr, epochs, batch_size, device):
    """Train LinearHead or MLPHead with BCEWithLogitsLoss + AdamW."""
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_f1, best_state = 0.0, None
    patience, patience_counter = 10, 0

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        for xb, yb in loader:
            logits = head(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(X_train)

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val).cpu().numpy()
        val_metrics = evaluate_all(val_logits, y_val.cpu().numpy())

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}  loss={avg_loss:.4f}  "
                f"val_wf1={val_metrics['weighted_f1']:.4f}  "
                f"val_mse={val_metrics['mse']:.4f}"
            )

        if val_metrics["weighted_f1"] > best_f1:
            best_f1 = val_metrics["weighted_f1"]
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  early stopping at epoch {epoch}")
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def train_ridge(splits):
    """Train RidgeHead with closed-form solution + lambda search."""
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    ridge = RidgeHead()
    best_lam, best_mse = ridge.search_lambda(X_train, y_train, X_val, y_val)
    print(f"  Ridge best lambda={best_lam}, val MSE={best_mse:.6f}")
    return ridge


def main():
    parser = argparse.ArgumentParser(description="MVP-2: Train baseline heads")
    parser.add_argument("--features_dir", type=str, default="features/phenotype")
    parser.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp", "ridge", "all"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = load_features(args.features_dir, device)
    in_dim = splits["train"][0].shape[1]
    out_dim = splits["train"][1].shape[1]
    print(f"Feature dim: {in_dim}, Output dim: {out_dim}")

    X_test, y_test = splits["test"]
    heads_to_run = [args.head] if args.head != "all" else ["linear", "mlp", "ridge"]

    for head_name in heads_to_run:
        print(f"\n--- Training: {head_name} ---")

        if head_name == "linear":
            head = LinearHead(in_dim, out_dim)
            head = train_neural_head(head, splits, args.lr, args.epochs, args.batch_size, device)
            head.eval()
            with torch.no_grad():
                test_preds = head(X_test).cpu().numpy()

        elif head_name == "mlp":
            head = MLPHead(in_dim, out_dim)
            head = train_neural_head(head, splits, args.lr, args.epochs, args.batch_size, device)
            head.eval()
            with torch.no_grad():
                test_preds = head(X_test).cpu().numpy()

        elif head_name == "ridge":
            ridge = train_ridge(splits)
            test_preds = ridge.predict(X_test).cpu().numpy()

        else:
            raise ValueError(f"Unknown head: {head_name}")

        test_labels = y_test.cpu().numpy()
        metrics = evaluate_all(test_preds, test_labels)
        print_metrics(metrics, header=f"Test results — {head_name}")


if __name__ == "__main__":
    main()
