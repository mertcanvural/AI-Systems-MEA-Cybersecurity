#!/usr/bin/env python
"""
Train SEAT similarity encoder for defense against model extraction attacks.

This script trains a similarity encoder using contrastive loss and adversarial
training as described in the SEAT paper.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.defense.SEAT.encoder import (
    SimilarityEncoder,
    contrastive_loss,
    adversarial_augmentation,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train SEAT similarity encoder for model extraction defense"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to defend"
    )
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="defense_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=256, help="Dimension of model embeddings"
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=128,
        help="Dimension of similarity encoder output",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
        help="Epsilon for adversarial training",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=5.0,
        help="Margin for contrastive loss (increased from paper default)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    from show_recommendations import load_model

    model = load_model(args.model_path, 3953, args.embedding_dim, device)

    # Create and initialize similarity encoder
    similarity_encoder = SimilarityEncoder(
        embedding_dim=args.embedding_dim, output_dim=args.output_dim
    )

    # Initialize with model weights if possible
    try:
        # Try to use the model's item embedding layer weights as initialization
        with torch.no_grad():
            similarity_encoder.encoder[0].weight.data = (
                model.item_embeddings.weight.data[:512, :]
            )
    except:
        print("Could not initialize from model weights, using random initialization")

    similarity_encoder.to(device)

    # Load training data
    print("Loading training data...")
    train_data = torch.load(args.train_data)

    # Extract embeddings from training data
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(train_data, batch_size=args.batch_size)):
            # Handle different data formats that might come from the DataLoader
            if isinstance(batch, tuple) or isinstance(batch, list):
                # If batch is a tuple/list, first element is usually item IDs
                item_ids = batch[0]
            else:
                # Otherwise use the batch directly
                item_ids = batch

            # Ensure it's on the correct device
            item_ids = item_ids.to(device)

            # Get embeddings
            item_embs = model.item_embeddings(item_ids)
            embeddings.append(item_embs.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Extracted {len(embeddings)} embeddings")

    # Create dataset with all possible triplets
    print("Creating training triplets...")

    # We'll create triplets on the fly during training
    # Just create a dataset of all embeddings for now
    dataset = TensorDataset(torch.arange(len(embeddings)))

    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer
    optimizer = optim.Adam(
        similarity_encoder.parameters(), lr=args.lr, weight_decay=1e-5
    )

    # Learning rate scheduler - reduce LR when plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Training
        similarity_encoder.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader)
        for (indices,) in progress_bar:
            # Get anchor embeddings
            anchor_indices = indices
            anchor_embs = embeddings[anchor_indices].to(device)

            # Generate positive samples via adversarial perturbation
            # Don't use no_grad here since we need gradients for adversarial generation
            positive_embs = adversarial_augmentation(
                model=similarity_encoder,
                embeddings=anchor_embs,
                epsilon=args.epsilon,
                alpha=args.epsilon / 10,
                steps=10,
            )

            # Generate negative samples by random selection
            with torch.no_grad():
                # (making sure they're different from anchor)
                negative_indices = torch.randint(
                    0, len(embeddings), (len(anchor_indices),)
                )
                for i, (anchor_idx, neg_idx) in enumerate(
                    zip(anchor_indices, negative_indices)
                ):
                    if anchor_idx == neg_idx:
                        # If same, pick another random index
                        negative_indices[i] = (neg_idx + 1) % len(embeddings)

                negative_embs = embeddings[negative_indices].to(device)

            # Forward pass
            anchor_out = similarity_encoder(anchor_embs)
            positive_out = similarity_encoder(positive_embs)
            negative_out = similarity_encoder(negative_embs)

            # Calculate loss
            loss = contrastive_loss(
                anchor=anchor_out,
                positive=positive_out,
                negative=negative_out,
                margin=args.margin,
            )

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.epochs} Loss: {loss.item():.6f}"
            )

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        similarity_encoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for (indices,) in val_loader:
                # Get anchor embeddings
                anchor_indices = indices
                anchor_embs = embeddings[anchor_indices].to(device)

                # For validation, we can use simpler adversarial examples or just slight perturbations
                # We can't use adversarial_augmentation here because we're in no_grad context
                # Just add a small random perturbation instead
                positive_embs = (
                    anchor_embs + torch.randn_like(anchor_embs) * args.epsilon * 0.1
                )

                negative_indices = torch.randint(
                    0, len(embeddings), (len(anchor_indices),)
                )
                for i, (anchor_idx, neg_idx) in enumerate(
                    zip(anchor_indices, negative_indices)
                ):
                    if anchor_idx == neg_idx:
                        negative_indices[i] = (neg_idx + 1) % len(embeddings)

                negative_embs = embeddings[negative_indices].to(device)

                # Forward pass
                anchor_out = similarity_encoder(anchor_embs)
                positive_out = similarity_encoder(positive_embs)
                negative_out = similarity_encoder(negative_embs)

                # Calculate loss
                loss = contrastive_loss(
                    anchor=anchor_out,
                    positive=positive_out,
                    negative=negative_out,
                    margin=args.margin,
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, "seat_encoder.pt")
            torch.save(
                {
                    "state_dict": similarity_encoder.state_dict(),
                    "config": {
                        "embedding_dim": args.embedding_dim,
                        "output_dim": args.output_dim,
                        "epsilon": args.epsilon,
                        "margin": args.margin,
                    },
                },
                save_path,
            )
            print(f"Saved best model to {save_path}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "figures", "training_loss.png"))

    print("Training complete!")


if __name__ == "__main__":
    main()
