import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SequentialRecommendationDataset(Dataset):
    """Dataset for sequential recommendation"""

    def __init__(self, sequences, targets, metadata=None, max_seq_length=50):
        """
        Initialize dataset

        Args:
            sequences: list of item sequences
            targets: list of target items
            metadata: dictionary of additional metadata features per item (optional)
            max_seq_length: maximum sequence length after padding/truncation
        """
        self.sequences = sequences
        self.targets = targets
        self.metadata = metadata
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]

        # Truncate or pad sequence
        if len(sequence) > self.max_seq_length:
            # Truncate to max_seq_length (keep most recent)
            sequence = sequence[-self.max_seq_length :]
        elif len(sequence) < self.max_seq_length:
            # Pad with zeros at the beginning
            sequence = [0] * (self.max_seq_length - len(sequence)) + sequence

        # Basic input is just the sequence
        input_dict = {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long),
        }

        # Add metadata if available
        if self.metadata is not None:
            # Extract metadata for sequence items
            for meta_name, meta_features in self.metadata.items():
                if isinstance(meta_features, dict):
                    # Item-level metadata (e.g., genres)
                    seq_meta = []
                    for item_id in sequence:
                        # Use zero vector for padding
                        if item_id == 0 or item_id not in meta_features:
                            # Create zero vector of appropriate size
                            feature_size = next(iter(meta_features.values())).shape[0]
                            seq_meta.append(np.zeros(feature_size, dtype=np.float32))
                        else:
                            seq_meta.append(meta_features[item_id])

                    input_dict[f"{meta_name}_features"] = torch.tensor(
                        np.array(seq_meta), dtype=torch.float32
                    )

        return input_dict


def load_movielens(file_path, movies_path=None, min_sequence_length=5, min_rating=3.5):
    """
    Load MovieLens dataset and convert to sequences

    Args:
        file_path: path to ratings dataset file
        movies_path: path to movies.dat file for metadata (optional)
        min_sequence_length: minimum number of interactions to keep a user
        min_rating: minimum rating to consider as positive interaction

    Returns:
        data: dictionary with user_sequences, num_items, and optional metadata
    """
    print(f"Loading MovieLens dataset from {file_path}")

    # Parse the file format
    if file_path.endswith(".dat"):
        # MovieLens 1M format
        df = pd.read_csv(
            file_path,
            sep="::",
            header=None,
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
        )
    else:
        # CSV format
        df = pd.read_csv(file_path)

    # Filter by rating if specified
    if min_rating is not None:
        print(f"Filtering ratings >= {min_rating}")
        df = df[df["rating"] >= min_rating]

    # Sort by user and timestamp
    df = df.sort_values(["userId", "timestamp"])

    # Create user sequences
    user_sequences = {}

    print("Creating user sequences:")
    for user_id, group in tqdm(df.groupby("userId")):
        item_ids = group["movieId"].values.tolist()

        # Only keep users with minimum sequence length
        if len(item_ids) >= min_sequence_length:
            user_sequences[user_id] = item_ids

    print(f"Created {len(user_sequences)} user sequences")

    # Get all unique items
    all_items = set()
    for items in user_sequences.values():
        all_items.update(items)

    num_items = max(all_items) + 1  # add 1 for padding/unknown

    # Load movie metadata if provided
    metadata = None
    if movies_path is not None and os.path.exists(movies_path):
        print(f"Loading movie metadata from {movies_path}")
        metadata = {}

        # Extract genres
        genres_set = set()
        movie_genres = {}

        with open(movies_path, "r", encoding="iso-8859-1") as f:
            for line in f:
                parts = line.strip().split("::")
                movie_id = int(parts[0])
                genres = parts[2].split("|")

                # Skip movies not in our dataset
                if movie_id not in all_items:
                    continue

                movie_genres[movie_id] = genres
                genres_set.update(genres)

        genres_list = sorted(list(genres_set))

        # Create genre vectors
        genre_vectors = {}
        for movie_id, genres in movie_genres.items():
            genre_vec = np.zeros(len(genres_list), dtype=np.float32)
            for i, genre in enumerate(genres_list):
                if genre in genres:
                    genre_vec[i] = 1.0
            genre_vectors[movie_id] = genre_vec

        metadata["genre"] = genre_vectors
        metadata["genre_names"] = genres_list

        print(f"Extracted genre features with {len(genres_list)} genres")

    return {
        "user_sequences": user_sequences,
        "num_items": num_items,
        "metadata": metadata,
    }


def create_train_val_test_splits(user_sequences, val_ratio=0.1, test_ratio=0.1):
    """
    Create train/validation/test splits from user sequences

    Args:
        user_sequences: dictionary mapping user_id to item sequence
        val_ratio: ratio of users to use for validation
        test_ratio: ratio of users to use for testing

    Returns:
        splits: dictionary with train/val/test sequences and targets
    """
    print("Creating dataset splits...")

    # Get all users
    users = list(user_sequences.keys())
    np.random.shuffle(users)

    # Determine split points
    num_users = len(users)
    num_val = int(num_users * val_ratio)
    num_test = int(num_users * test_ratio)

    # Split users
    val_users = users[:num_val]
    test_users = users[num_val : num_val + num_test]
    train_users = users[num_val + num_test :]

    # Function to create input/target pairs
    def create_sequences_and_targets(user_subset):
        sequences = []
        targets = []

        for user_id in user_subset:
            sequence = user_sequences[user_id]

            # Use last item as target, rest as input
            if len(sequence) > 1:
                sequences.append(sequence[:-1])
                targets.append(sequence[-1])

        return sequences, targets

    # Create splits
    train_sequences, train_targets = create_sequences_and_targets(train_users)
    val_sequences, val_targets = create_sequences_and_targets(val_users)
    test_sequences, test_targets = create_sequences_and_targets(test_users)

    return {
        "train_sequences": train_sequences,
        "train_targets": train_targets,
        "val_sequences": val_sequences,
        "val_targets": val_targets,
        "test_sequences": test_sequences,
        "test_targets": test_targets,
    }


def create_data_loaders(
    splits, metadata=None, batch_size=32, max_seq_length=50, num_workers=0
):
    """
    Create dataloader objects for all splits

    Args:
        splits: dictionary with train/val/test sequences and targets
        metadata: dictionary with metadata features (optional)
        batch_size: batch size for dataloader
        max_seq_length: maximum sequence length
        num_workers: number of workers for dataloader

    Returns:
        dataloaders: dictionary with train/val/test dataloaders
    """
    # Create datasets
    train_dataset = SequentialRecommendationDataset(
        splits["train_sequences"],
        splits["train_targets"],
        metadata=metadata,
        max_seq_length=max_seq_length,
    )

    val_dataset = SequentialRecommendationDataset(
        splits["val_sequences"],
        splits["val_targets"],
        metadata=metadata,
        max_seq_length=max_seq_length,
    )

    test_dataset = SequentialRecommendationDataset(
        splits["test_sequences"],
        splits["test_targets"],
        metadata=metadata,
        max_seq_length=max_seq_length,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
