"""MVP-1: Extract pooled embeddings from a frozen BioClinical ModernBERT encoder.

Usage:
    python -m chwls.extract_features \
        --model thomas-sounack/BioClinical-ModernBERT-base \
        --dataset Phenotype \
        --batch_size 8 \
        --output_dir features/phenotype
"""

import argparse
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataloader.dataloader import get_data


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


@torch.no_grad()
def extract_split(model, tokenizer, texts, labels_list, batch_size, max_length, device):
    all_features = []
    all_labels = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_labels = labels_list[start : start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)
        pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

        all_features.append(pooled.cpu())
        all_labels.append(torch.tensor(batch_labels, dtype=torch.float32))

        if (start // batch_size) % 50 == 0:
            print(f"  processed {start + len(batch_texts)}/{len(texts)}")

    X = torch.cat(all_features, dim=0)
    y = torch.cat(all_labels, dim=0)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="MVP-1: Extract features from a frozen encoder")
    parser.add_argument("--model", type=str, default="thomas-sounack/BioClinical-ModernBERT-base")
    parser.add_argument("--dataset", type=str, default="Phenotype")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="features/phenotype")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    max_length = tokenizer.model_max_length if tokenizer.model_max_length < 10000 else 512
    print(f"Max sequence length: {max_length}")

    print(f"Loading dataset: {args.dataset}")
    data_wrapper = get_data(args.dataset)
    ds = data_wrapper.dataset

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, hf_split_name in [("train", "train"), ("val", "validation"), ("test", "test")]:
        split = ds[hf_split_name]
        texts = split["text"]
        labels = split["labels"]

        print(f"Extracting {split_name}: {len(texts)} samples ...")
        X, y = extract_split(model, tokenizer, texts, labels, args.batch_size, max_length, device)
        print(f"  X shape: {X.shape}, y shape: {y.shape}")

        out_path = os.path.join(args.output_dir, f"{split_name}.pt")
        torch.save({"X": X, "y": y}, out_path)
        print(f"  saved -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
