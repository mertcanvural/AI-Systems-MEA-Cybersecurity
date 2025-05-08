#!/usr/bin/env python
"""
Prepare data for SEAT defense training and evaluation.

This script prepares training data for the SEAT similarity encoder
and preprocesses attack and benign query data for evaluation.
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Prepare data for SEAT defense")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the target model"
    )
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument("--attack_data", type=str, help="Path to attack query data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="defense_data",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for processing"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    from show_recommendations import load_model

    model = load_model(args.model_path, 3953, 256, device)

    # Load training data
    print("Loading training data...")
    train_data = torch.load(args.train_data)

    # Process and split training data
    print("Processing training data...")

    # Split train/val/test
    train_size = int(0.8 * len(train_data))
    val_size = int(0.1 * len(train_data))
    test_size = len(train_data) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        train_data, [train_size, val_size, test_size]
    )

    # Save processed training data
    train_path = os.path.join(args.output_dir, "seat_train_data.pt")
    val_path = os.path.join(args.output_dir, "seat_val_data.pt")
    test_path = os.path.join(args.output_dir, "seat_test_data.pt")

    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    torch.save(test_dataset, test_path)

    print(f"Saved training data: {train_size} samples")
    print(f"Saved validation data: {val_size} samples")
    print(f"Saved test data: {test_size} samples")

    # Process attack data if provided
    if args.attack_data:
        print("\nProcessing attack data...")
        attack_data = torch.load(args.attack_data)

        # Save processed attack data
        attack_path = os.path.join(args.output_dir, "seat_attack_data.pt")
        torch.save(attack_data, attack_path)
        print(f"Saved attack data: {len(attack_data)} samples")

    # Generate sample encodings for quick testing
    print("\nGenerating sample encodings...")

    # Sample data for quick testing
    sample_size = min(1000, len(test_dataset))
    sample_indices = torch.randperm(len(test_dataset))[:sample_size]
    sample_data = [test_dataset[i] for i in sample_indices]

    # Extract item IDs for samples
    sample_items = []
    for item in sample_data:
        if isinstance(item, tuple):
            sample_items.append(item[0])  # Assume first element is item ID
        else:
            sample_items.append(item)

    # Get embeddings for sample data
    sample_embeddings = []

    with torch.no_grad():
        # Process in batches
        batch_size = args.batch_size
        for i in range(0, len(sample_items), batch_size):
            items_batch = sample_items[i : min(i + batch_size, len(sample_items))]

            # Stack tensors or convert list to tensor
            if all(isinstance(item, torch.Tensor) for item in items_batch):
                batch = torch.stack(items_batch).to(device)
            else:
                # Try to convert to tensors if they aren't already
                tensor_batch = []
                for item in items_batch:
                    if isinstance(item, torch.Tensor):
                        tensor_batch.append(item)
                    elif isinstance(item, (int, float)):
                        tensor_batch.append(torch.tensor([item]))
                    else:
                        # Skip items we can't convert
                        print(f"Warning: Skipping item of type {type(item)}")

                if tensor_batch:
                    batch = torch.stack(tensor_batch).to(device)
                else:
                    # Skip this batch if all items were skipped
                    continue

            # Extract embeddings
            embedding = model.item_embeddings(batch).cpu()
            sample_embeddings.append(embedding)

    # Combine batches
    sample_embeddings = torch.cat(sample_embeddings, dim=0)

    # Save sample embeddings
    sample_path = os.path.join(args.output_dir, "seat_sample_embeddings.pt")
    torch.save(sample_embeddings, sample_path)

    print(f"Saved sample embeddings: {len(sample_embeddings)} samples")
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
