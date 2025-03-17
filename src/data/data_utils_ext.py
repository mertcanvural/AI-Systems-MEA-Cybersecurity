import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from src.data.data_utils import load_movielens, create_train_val_test_splits


class SequentialRecommendationDatasetExt(Dataset):
    """enhanced dataset for sequential recommendation with metadata"""

    def __init__(self, sequences, targets, metadata=None, max_seq_length=50):
        """
        initialize enhanced dataset

        args:
            sequences: list of item sequences
            targets: list of target items
            metadata: dictionary of additional metadata features per item
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

        # truncate or pad sequence
        if len(sequence) > self.max_seq_length:
            # truncate to max_seq_length (keep most recent)
            sequence = sequence[-self.max_seq_length :]
        elif len(sequence) < self.max_seq_length:
            # pad with zeros at the beginning
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


def load_movielens_enhanced(
    file_path, movies_path=None, min_sequence_length=5, min_rating=3.5
):
    """
    load movielens dataset with enhanced features

    args:
        file_path: path to movielens csv file
        movies_path: path to movies.dat file for metadata
        min_sequence_length: minimum number of interactions to keep a user
        min_rating: minimum rating to consider as positive interaction

    returns:
        data: dictionary with enhanced user_sequences and metadata
    """
    print(f"loading enhanced movielens dataset from {file_path}")

    # read csv file
    df = pd.read_csv(file_path)

    # filter by rating
    if min_rating is not None:
        print(f"filtering ratings >= {min_rating}")
        df = df[df["rating"] >= min_rating]

    # sort by user and timestamp
    df = df.sort_values(["userId", "timestamp"])

    # create user sequences
    user_sequences = {}

    print("creating user sequences:")
    for user_id, group in tqdm(df.groupby("userId")):
        item_ids = group["movieId"].values.tolist()

        # only keep users with minimum sequence length
        if len(item_ids) >= min_sequence_length:
            user_sequences[user_id] = item_ids

    print(f"created {len(user_sequences)} user sequences")

    # get all unique items
    all_items = set()
    for items in user_sequences.values():
        all_items.update(items)

    num_items = max(all_items) + 1  # add 1 for padding/unknown

    # Load movie metadata if provided
    metadata = None
    if movies_path is not None and os.path.exists(movies_path):
        print(f"loading movie metadata from {movies_path}")
        metadata = {}

        # Extract genres
        genres_set = set()
        movie_genres = {}

        with open(movies_path, "r", encoding="iso-8859-1") as f:
            for line in f:
                parts = line.strip().split("::")
                movie_id = int(parts[0])
                title = parts[1]
                genres = parts[2].split("|")

                # Skip movies not in our dataset
                if movie_id not in all_items:
                    continue

                movie_genres[movie_id] = genres
                genres_set.update(genres)

        genres_list = sorted(list(genres_set))
        genres_to_idx = {genre: i for i, genre in enumerate(genres_list)}

        # Create genre vectors
        genre_vectors = {}
        for movie_id, genres in movie_genres.items():
            genre_vec = np.zeros(len(genres_list), dtype=np.float32)
            for genre in genres:
                genre_vec[genres_to_idx[genre]] = 1.0
            genre_vectors[movie_id] = genre_vec

        metadata["genre"] = genre_vectors
        metadata["genre_names"] = genres_list

        print(f"extracted genre features with {len(genres_list)} genres")

    return {
        "user_sequences": user_sequences,
        "num_items": num_items,
        "metadata": metadata,
    }


def create_data_loaders_enhanced(
    splits, metadata=None, batch_size=32, max_seq_length=50, num_workers=0
):
    """
    create enhanced dataloader objects for all splits

    args:
        splits: dictionary with train/val/test sequences and targets
        metadata: dictionary with metadata features
        batch_size: batch size for dataloader
        max_seq_length: maximum sequence length
        num_workers: number of workers for dataloader

    returns:
        dataloaders: dictionary with train/val/test dataloaders
    """
    # create datasets
    train_dataset = SequentialRecommendationDatasetExt(
        splits["train_sequences"],
        splits["train_targets"],
        metadata=metadata,
        max_seq_length=max_seq_length,
    )

    val_dataset = SequentialRecommendationDatasetExt(
        splits["val_sequences"],
        splits["val_targets"],
        metadata=metadata,
        max_seq_length=max_seq_length,
    )

    test_dataset = SequentialRecommendationDatasetExt(
        splits["test_sequences"],
        splits["test_targets"],
        metadata=metadata,
        max_seq_length=max_seq_length,
    )

    # create dataloaders
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
