import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_utils import load_movielens
from src.data.preprocessing import extract_item_features, analyze_dataset
from src.data.visualization import (
    plot_sequence_length_distribution,
    plot_item_popularity,
    plot_genre_distribution,
    plot_year_distribution,
    plot_user_item_matrix,
)


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="analyze movielens dataset")

    parser.add_argument(
        "--data_dir", type=str, default="data", help="directory containing the datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
        choices=["ml-1m", "ml-20m"],
        help="dataset to use",
    )
    parser.add_argument(
        "--output_dir", type=str, default="figures", help="directory to save figures"
    )
    parser.add_argument(
        "--min_seq_length", type=int, default=5, help="minimum sequence length to keep"
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data_path = os.path.join(args.data_dir, f"{args.dataset}.csv")
    print(f"loading data from {data_path}...")
    data = load_movielens(data_path, min_sequence_length=args.min_seq_length)
    user_sequences = data["user_sequences"]

    # Load movie features
    movies_path = os.path.join(args.data_dir, args.dataset, "movies.dat")
    print(f"loading movie features from {movies_path}...")
    item_features, genres_list = extract_item_features(movies_path)

    # Analyze dataset
    print("analyzing dataset...")
    stats = analyze_dataset(user_sequences, item_features)

    # Print statistics
    print("\ndataset statistics:")
    print(f"number of users: {stats['num_users']}")
    print(f"number of items: {stats['num_items']}")
    print(f"average sequence length: {stats['seq_length_mean']:.2f}")
    print(f"median sequence length: {stats['seq_length_median']:.2f}")
    print(
        f"min/max sequence length: {stats['seq_length_min']}/{stats['seq_length_max']}"
    )
    print(f"average item frequency: {stats['item_count_mean']:.2f}")
    print(f"median item frequency: {stats['item_count_median']:.2f}")
    print(
        f"min/max item frequency: {stats['item_count_min']}/{stats['item_count_max']}"
    )

    print("\ntop 10 most popular movies:")
    for i, (movie_id, count) in enumerate(stats["top_items"], 1):
        title = item_features.get(movie_id, {}).get("title", "Unknown")
        print(f"{i}. {title} (ID: {movie_id}) - {count} interactions")

    print("\ntop 10 most common genres:")
    for i, (genre, count) in enumerate(stats["genre_counts"].most_common(10), 1):
        print(f"{i}. {genre} - {count} occurrences")

    # Create visualizations
    print("\ngenerating visualizations...")

    # 1. Sequence length distribution
    plot_sequence_length_distribution(
        user_sequences,
        save_path=os.path.join(args.output_dir, "sequence_length_dist.png"),
    )

    # 2. Item popularity
    plot_item_popularity(
        user_sequences,
        top_n=20,
        save_path=os.path.join(args.output_dir, "item_popularity.png"),
    )

    # 3. Genre distribution
    plot_genre_distribution(
        item_features,
        user_sequences=user_sequences,
        save_path=os.path.join(args.output_dir, "genre_distribution.png"),
    )

    # 4. Year distribution
    plot_year_distribution(
        item_features,
        user_sequences=user_sequences,
        save_path=os.path.join(args.output_dir, "year_distribution.png"),
    )

    # 5. User-item matrix
    plot_user_item_matrix(
        user_sequences,
        max_users=50,
        max_items=100,
        save_path=os.path.join(args.output_dir, "user_item_matrix.png"),
    )

    print(f"analysis complete! figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
