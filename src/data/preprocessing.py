import os
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm


def preprocess_movielens(df, min_rating=3.5, min_interactions=5):
    """
    preprocess movielens dataset with filtering

    args:
        df: pandas dataframe with movielens data
        min_rating: minimum rating to consider as positive interaction
        min_interactions: minimum interactions per user to keep

    returns:
        processed_df: filtered and processed dataframe
    """
    print("preprocessing movielens dataset...")

    # Convert to positive-only interactions (implicit feedback)
    print(f"filtering ratings >= {min_rating} as positive interactions")
    positive_df = df[df["rating"] >= min_rating].copy()

    # Count interactions per user
    user_counts = positive_df["userId"].value_counts()

    # Filter users with too few interactions
    valid_users = user_counts[user_counts >= min_interactions].index
    print(
        f"keeping {len(valid_users)}/{len(user_counts)} users with >= {min_interactions} interactions"
    )

    filtered_df = positive_df[positive_df["userId"].isin(valid_users)]

    # Sort by user and timestamp
    filtered_df = filtered_df.sort_values(["userId", "timestamp"])

    return filtered_df


def extract_item_features(movies_path):
    """
    extract movie features from movies.dat

    args:
        movies_path: path to movies.dat file

    returns:
        item_features: dictionary with movie features
    """
    item_features = {}
    genres_set = set()

    # Read movies file
    with open(movies_path, "r", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.strip().split("::")
            movie_id = int(parts[0])
            title = parts[1]
            genres = parts[2].split("|")

            # Extract year from title if present
            year = None
            if title.endswith(")") and "(" in title:
                try:
                    year_str = title.split("(")[-1].split(")")[0]
                    if year_str.isdigit():
                        year = int(year_str)
                except:
                    pass

            # Store features
            item_features[movie_id] = {"title": title, "genres": genres, "year": year}

            # Update genres set
            genres_set.update(genres)

    # Convert genres to one-hot encoding
    genres_list = sorted(list(genres_set))
    for movie_id, features in item_features.items():
        genre_vector = [
            1 if genre in features["genres"] else 0 for genre in genres_list
        ]
        item_features[movie_id]["genre_vector"] = np.array(
            genre_vector, dtype=np.float32
        )

    print(
        f"extracted features for {len(item_features)} movies with {len(genres_list)} genres"
    )
    return item_features, genres_list


def analyze_dataset(user_sequences, item_features=None):
    """
    analyze dataset statistics

    args:
        user_sequences: dictionary mapping user_id to item sequence
        item_features: dictionary with item features

    returns:
        stats: dictionary with statistics
    """
    stats = {}

    # Sequence length statistics
    seq_lengths = [len(seq) for seq in user_sequences.values()]
    stats["num_users"] = len(user_sequences)
    stats["seq_length_mean"] = np.mean(seq_lengths)
    stats["seq_length_std"] = np.std(seq_lengths)
    stats["seq_length_min"] = min(seq_lengths)
    stats["seq_length_max"] = max(seq_lengths)
    stats["seq_length_median"] = np.median(seq_lengths)

    # Item popularity
    all_items = [item for seq in user_sequences.values() for item in seq]
    item_counts = Counter(all_items)
    stats["num_items"] = len(item_counts)
    stats["item_count_mean"] = np.mean(list(item_counts.values()))
    stats["item_count_std"] = np.std(list(item_counts.values()))
    stats["item_count_min"] = min(item_counts.values())
    stats["item_count_max"] = max(item_counts.values())
    stats["item_count_median"] = np.median(list(item_counts.values()))

    # Top items
    top_items = item_counts.most_common(10)
    stats["top_items"] = top_items

    # Genre analysis if item features provided
    if item_features:
        genre_counts = Counter()
        for user_id, sequence in user_sequences.items():
            for item in sequence:
                if item in item_features:
                    genre_counts.update(item_features[item]["genres"])

        stats["genre_counts"] = genre_counts

    return stats


def create_negative_samples(user_sequences, num_items, num_neg=5, seed=42):
    """
    create negative samples for each user sequence

    args:
        user_sequences: dictionary mapping user_id to item sequence
        num_items: number of unique items
        num_neg: number of negative samples per positive
        seed: random seed

    returns:
        negative_samples: dictionary mapping user_id to list of negative items
    """
    np.random.seed(seed)
    negative_samples = {}

    all_items = set(range(1, num_items))  # Exclude padding token 0

    for user_id, sequence in tqdm(
        user_sequences.items(), desc="creating negative samples"
    ):
        # Get user's positive items
        positive_items = set(sequence)

        # Get potential negative items
        negative_candidates = list(all_items - positive_items)

        # Sample negative items
        if len(negative_candidates) >= num_neg:
            negatives = np.random.choice(negative_candidates, num_neg, replace=False)
        else:
            # If not enough candidates, sample with replacement
            negatives = np.random.choice(negative_candidates, num_neg, replace=True)

        negative_samples[user_id] = negatives.tolist()

    return negative_samples
