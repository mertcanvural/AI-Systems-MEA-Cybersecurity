import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import torch.nn as nn

from src.models.base_model import SimpleSequentialRecommender
from src.data.data_utils import (
    load_movielens,
    create_train_val_test_splits,
    create_data_loaders,
)


def evaluate(model, data_loader, device):
    """Evaluate model on data loader"""
    model.eval()
    total_loss = 0
    hits = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            # Hit@10 metric
            _, top_indices = torch.topk(logits, k=10, dim=1)
            for i, label in enumerate(labels):
                if label in top_indices[i]:
                    hits += 1
                total += 1

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    hit_rate = hits / total if total > 0 else 0

    return avg_loss, hit_rate


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=64, help="Dimension of embeddings"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to save model",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Disable early stopping and train for all epochs",
    )
    return parser.parse_args()


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=0.001,
    device=None,
    model_path="checkpoints/best_model.pt",
    disable_early_stopping=False,
):
    """Train model on MovieLens dataset"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model, optimizer, criterion
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 5
    patience_counter = 0
    epochs_without_improvement = 0

    # Store metrics for visualization
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_hit_rate": [],
        "val_hit_rate": [],
    }

    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(
        f"Training model with embedding dimension {model.embedding_dim} for {num_epochs} epochs"
    )
    print(f"Batch size: {train_loader.batch_size}, Learning rate: {lr}")

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        train_hits = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)

            # Add L2 regularization
            l2_reg = 0.01
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_reg * 1e-4

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_loss += loss.item()

            # Hit@10 metric
            _, top_indices = torch.topk(logits, k=10, dim=1)
            for i, label in enumerate(labels):
                if label in top_indices[i]:
                    train_hits += 1
                train_total += 1

        # Evaluation
        val_loss, val_hit_rate = evaluate(model, val_loader, device)
        avg_train_loss = train_loss / len(train_loader)
        train_hit_rate = train_hits / train_total if train_total > 0 else 0

        # Store metrics
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_hit_rate"].append(train_hit_rate)
        metrics["val_hit_rate"].append(val_hit_rate)

        # Print metrics
        print(
            f"Epoch {epoch}/{num_epochs} - Train loss: {avg_train_loss:.4f}, "
            f"Val loss: {val_loss:.4f}, Train Hit@10: {train_hit_rate:.4f}, "
            f"Val Hit@10: {val_hit_rate:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_hit_rate": val_hit_rate,
            }
            torch.save(checkpoint, model_path)
            print(f"  Saved best model at epoch {epoch}")
        else:
            patience_counter += 1
            epochs_without_improvement += 1

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Early stopping
        if not disable_early_stopping and patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break

    # Save metrics for visualization
    metrics_path = os.path.join(os.path.dirname(model_path), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    print(
        f"Best model at epoch {best_epoch}/{epoch} with validation loss {best_val_loss:.4f}"
    )
    print("Training complete!")
    print("You can now run evaluate_recommendations.py to test the model.")

    return model, metrics


def main(args=None):
    """Main function"""
    if args is None:
        args = parse_args()

    # Load data
    data = load_movielens("data/ml-1m/ratings.dat", min_rating=4.0)
    print(
        f"Loaded {len(data['user_sequences'])} user sequences with {data['num_items']} unique items"
    )

    # Create dataset splits
    print("Creating dataset splits...")
    splits = create_train_val_test_splits(data["user_sequences"])
    loaders = create_data_loaders(splits, batch_size=args.batch_size)

    # Create model
    model = SimpleSequentialRecommender(
        data["num_items"], embedding_dim=args.embedding_dim, dropout_rate=args.dropout
    )
    print(f"Model architecture: {model}")

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training model on {device}...")

    model, metrics = train_model(
        model,
        loaders["train"],
        loaders["val"],
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        model_path=args.model_path,
        disable_early_stopping=args.disable_early_stopping,
    )

    return model, metrics


if __name__ == "__main__":
    main()
