import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse
import pandas as pd
import seaborn as sns
from collections import defaultdict

from src.models.base_model import SimpleSequentialRecommender
from src.data.data_utils import load_movielens


def load_model(model_path, num_items, embedding_dim, device=None):
    """Load a model from checkpoint"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleSequentialRecommender(num_items, embedding_dim)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e


def generate_test_sequences(num_sequences=50, seq_length=5, num_items=3953, seed=42):
    """Generate random test sequences"""
    np.random.seed(seed)
    sequences = []

    for _ in range(num_sequences):
        length = np.random.randint(1, seq_length + 1)
        sequence = np.random.randint(1, num_items, size=length).tolist()
        sequences.append(sequence)

    return sequences


def get_model_recommendations(model, sequence, k=10, device=None):
    """Get model recommendations for a sequence"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
        logits = model(sequence_tensor)
        scores, indices = torch.topk(logits, k=k, dim=1)

        # Convert to list of item IDs
        recommendations = indices[0].tolist()

    return recommendations


def compare_recommendations(
    target_model, surrogate_model, sequences, k_values=[1, 5, 10, 20], device=None
):
    """Compare recommendations from target and surrogate models"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize metrics
    overlaps = {k: [] for k in k_values}
    rank_correlations = []

    for sequence in tqdm(sequences, desc="Comparing models"):
        # Get target model recommendations
        target_recs = get_model_recommendations(
            target_model, sequence, k=max(k_values), device=device
        )

        # Get surrogate model recommendations
        surrogate_recs = get_model_recommendations(
            surrogate_model, sequence, k=max(k_values), device=device
        )

        # Calculate overlap at different k values
        for k in k_values:
            target_at_k = set(target_recs[:k])
            surrogate_at_k = set(surrogate_recs[:k])
            overlap = len(target_at_k.intersection(surrogate_at_k)) / k
            overlaps[k].append(overlap)

        # Calculate Spearman rank correlation
        target_ranks = {item: i for i, item in enumerate(target_recs)}
        surrogate_ranks = {item: i for i, item in enumerate(surrogate_recs)}

        # Find common items
        common_items = set(target_recs).intersection(set(surrogate_recs))

        if common_items:
            # Calculate differences in ranks
            rank_diffs = [
                (target_ranks[item] - surrogate_ranks[item]) ** 2
                for item in common_items
            ]
            n = len(common_items)

            # Spearman correlation
            if n > 1:  # Need at least 2 items to calculate correlation
                correlation = 1 - (6 * sum(rank_diffs) / (n * (n**2 - 1)))
                rank_correlations.append(correlation)

    # Calculate average metrics
    metrics = {}
    for k in k_values:
        metrics[f"overlap@{k}"] = np.mean(overlaps[k])

    if rank_correlations:
        metrics["rank_correlation"] = np.mean(rank_correlations)

    return metrics, overlaps


def create_overlap_boxplot(overlaps, output_path="attack_results/overlap_boxplot.png"):
    """Create boxplot of overlap metrics"""
    plt.figure(figsize=(12, 6))

    # Prepare data for boxplot
    data = []
    labels = []

    for k, values in sorted(overlaps.items()):
        data.append(values)
        labels.append(f"overlap@{k}")

    plt.boxplot(data, labels=labels)
    plt.ylabel("Overlap (proportion)")
    plt.title(
        "Distribution of Recommendation Overlap Between Target and Surrogate Models"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add mean values as text
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        plt.text(i + 1, 0.05, f"Mean: {mean_val:.3f}", ha="center")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"Saved boxplot to {output_path}")


def create_metrics_barchart(metrics, output_path="attack_results/metrics_barchart.png"):
    """Create bar chart of metrics"""
    # Filter out non-numeric metrics
    metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    plt.bar(x, list(metrics.values()), color="cornflowerblue")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Model Extraction Attack Performance")
    plt.xticks(x, list(metrics.keys()), rotation=45)

    for i, v in enumerate(list(metrics.values())):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"Saved bar chart to {output_path}")


def create_recommendation_comparison(
    target_model,
    surrogate_model,
    sequences,
    movies_data,
    output_path="attack_results/recommendation_examples.txt",
):
    """Create a text file with examples of recommendations from both models"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get movie titles
    movie_titles = {}
    for movie_id, title, genres in movies_data:
        movie_titles[movie_id] = title

    with open(output_path, "w") as f:
        f.write("RECOMMENDATION COMPARISON EXAMPLES\n")
        f.write("==================================\n\n")

        for i, sequence in enumerate(sequences[:5]):  # Use first 5 sequences
            f.write(f"Example {i+1}\n")
            f.write("---------\n")

            # Write sequence
            f.write("Input sequence:\n")
            for movie_id in sequence:
                title = movie_titles.get(movie_id, f"Unknown (ID: {movie_id})")
                f.write(f"  - {title} (ID: {movie_id})\n")

            # Get recommendations
            target_recs = get_model_recommendations(target_model, sequence, k=10)
            surrogate_recs = get_model_recommendations(surrogate_model, sequence, k=10)

            # Write target recommendations
            f.write("\nTarget Model Recommendations:\n")
            for rank, movie_id in enumerate(target_recs):
                title = movie_titles.get(movie_id, f"Unknown (ID: {movie_id})")
                f.write(f"  {rank+1}. {title} (ID: {movie_id})\n")

            # Write surrogate recommendations
            f.write("\nSurrogate Model Recommendations:\n")
            for rank, movie_id in enumerate(surrogate_recs):
                title = movie_titles.get(movie_id, f"Unknown (ID: {movie_id})")
                f.write(f"  {rank+1}. {title} (ID: {movie_id})\n")

            # Calculate and write overlap
            overlap = len(set(target_recs).intersection(set(surrogate_recs)))
            f.write(f"\nOverlap: {overlap}/10 items ({overlap/10:.1%})\n\n")

            f.write("\n" + "=" * 50 + "\n\n")

    print(f"Saved recommendation examples to {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare Target and Surrogate Models")
    parser.add_argument(
        "--target-model",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to target model checkpoint",
    )
    parser.add_argument(
        "--surrogate-model",
        type=str,
        default="checkpoints/surrogate_model.pt",
        help="Path to surrogate model checkpoint",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension for the models",
    )
    parser.add_argument(
        "--num-sequences", type=int, default=100, help="Number of test sequences"
    )
    parser.add_argument(
        "--seq-length", type=int, default=10, help="Maximum sequence length"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ml-1m",
        help="Path to MovieLens dataset directory",
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MovieLens data
    ratings_file = os.path.join(args.data_path, "ratings.dat")
    movies_file = os.path.join(args.data_path, "movies.dat")

    data = load_movielens(ratings_file)
    num_items = data["num_items"]

    # Load movie information
    movies_data = []
    with open(movies_file, "r", encoding="latin1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) == 3:
                movie_id = int(parts[0])
                title = parts[1]
                genres = parts[2].split("|")
                movies_data.append((movie_id, title, genres))

    print(f"Loaded {len(movies_data)} movies")

    # Load models
    print(f"Loading target model from {args.target_model}")
    target_model = load_model(args.target_model, num_items, args.embedding_dim, device)

    print(f"Loading surrogate model from {args.surrogate_model}")
    surrogate_model = load_model(
        args.surrogate_model, num_items, args.embedding_dim, device
    )

    # Generate test sequences
    test_sequences = generate_test_sequences(
        args.num_sequences, args.seq_length, num_items
    )
    print(f"Generated {len(test_sequences)} test sequences")

    # Compare models
    metrics, overlaps = compare_recommendations(
        target_model, surrogate_model, test_sequences
    )

    # Save metrics to JSON
    os.makedirs("attack_results", exist_ok=True)
    with open("attack_results/comparison_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics to attack_results/comparison_metrics.json")

    # Create visualizations
    create_metrics_barchart(metrics)
    create_overlap_boxplot(overlaps)

    # Create recommendation examples
    create_recommendation_comparison(
        target_model, surrogate_model, test_sequences, movies_data
    )

    print("Model comparison complete!")


if __name__ == "__main__":
    main()
