import os
import torch
import argparse
import numpy as np
from tabulate import tabulate

from src.models.base_model import SimpleSequentialRecommender
from src.data.data_utils import load_movielens


def load_model(model_path, num_items, embedding_dim, device=None):
    """Load model from checkpoint"""
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


def load_movie_data(movies_file):
    """Load movie data from file"""
    movies_data = {}
    try:
        with open(movies_file, "r", encoding="latin1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) == 3:
                    movie_id = int(parts[0])
                    title = parts[1]
                    genres = parts[2].split("|")
                    movies_data[movie_id] = {"title": title, "genres": genres}
    except Exception as e:
        print(f"Error loading movie data: {e}")

    return movies_data


def get_model_recommendations(model, sequence, k=10, device=None):
    """Get model recommendations for a sequence"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
        logits = model(sequence_tensor)
        scores, indices = torch.topk(logits, k=k, dim=1)

        # Convert to list of item IDs and scores
        recommendations = []
        for i in range(len(indices[0])):
            item_id = indices[0][i].item()
            score = scores[0][i].item()
            recommendations.append((item_id, score))

    return recommendations


def create_example_sequences(num_items, num_examples=5, seed=42):
    """Create example sequences for comparison"""
    np.random.seed(seed)

    example_sequences = [
        # Example 1: Short sequence of popular movies
        [260, 1196, 1210],  # Star Wars, Star Trek, Star Trek: First Contact
        # Example 2: Action movies
        [588, 589, 592, 593],  # Die Hard series
        # Example 3: Children's movies
        [595, 34, 314],  # Snow White, Babe, Aladdin
        # Example 4: Oscar winners
        [858, 318, 169],  # Godfather, Shawshank Redemption, Braveheart
        # Example 5: Random sequence
        np.random.randint(1, num_items, size=4).tolist(),
    ]

    return example_sequences


def calculate_similarity(target_recs, surrogate_recs):
    """Calculate similarity between recommendation lists"""
    target_ids = [item_id for item_id, _ in target_recs]
    surrogate_ids = [item_id for item_id, _ in surrogate_recs]

    # Calculate overlap
    overlap = len(set(target_ids).intersection(set(surrogate_ids)))
    overlap_percentage = overlap / len(target_ids) * 100

    # Calculate position-aware similarity (Kendall's tau)
    common_items = set(target_ids).intersection(set(surrogate_ids))
    if len(common_items) <= 1:
        kendall_tau = 0
    else:
        target_ranks = {item_id: i for i, item_id in enumerate(target_ids)}
        surrogate_ranks = {item_id: i for i, item_id in enumerate(surrogate_ids)}

        concordant = 0
        discordant = 0

        common_items_list = list(common_items)
        for i in range(len(common_items_list)):
            for j in range(i + 1, len(common_items_list)):
                item_i = common_items_list[i]
                item_j = common_items_list[j]

                target_rank_diff = target_ranks[item_i] - target_ranks[item_j]
                surrogate_rank_diff = surrogate_ranks[item_i] - surrogate_ranks[item_j]

                if (target_rank_diff * surrogate_rank_diff) > 0:
                    concordant += 1
                else:
                    discordant += 1

        if concordant + discordant > 0:
            kendall_tau = (concordant - discordant) / (concordant + discordant)
        else:
            kendall_tau = 0

    return {
        "overlap": overlap,
        "overlap_percentage": overlap_percentage,
        "kendall_tau": kendall_tau,
    }


def compare_examples(
    target_model, surrogate_model, sequences, movies_data, output_file=None, top_k=10
):
    """Compare recommendations for example sequences"""
    results = []

    for i, sequence in enumerate(sequences):
        # Convert sequence to movie titles for display
        sequence_titles = []
        for movie_id in sequence:
            if movie_id in movies_data:
                title = movies_data[movie_id]["title"]
                sequence_titles.append(f"{title} ({movie_id})")
            else:
                sequence_titles.append(f"Unknown ({movie_id})")

        sequence_str = " → ".join(sequence_titles)

        # Get recommendations from both models
        target_recs = get_model_recommendations(target_model, sequence, k=top_k)
        surrogate_recs = get_model_recommendations(surrogate_model, sequence, k=top_k)

        # Calculate similarity metrics
        similarity = calculate_similarity(target_recs, surrogate_recs)

        # Format recommendations as strings
        target_rec_strings = []
        surrogate_rec_strings = []

        for j in range(top_k):
            t_item_id, t_score = target_recs[j]
            s_item_id, s_score = surrogate_recs[j]

            t_title = (
                movies_data[t_item_id]["title"]
                if t_item_id in movies_data
                else "Unknown"
            )
            s_title = (
                movies_data[s_item_id]["title"]
                if s_item_id in movies_data
                else "Unknown"
            )

            t_str = f"{j+1}. {t_title} ({t_item_id}): {t_score:.3f}"
            s_str = f"{j+1}. {s_title} ({s_item_id}): {s_score:.3f}"

            # Highlight matches
            if t_item_id == s_item_id:
                t_str = f"* {t_str}"
                s_str = f"* {s_str}"

            target_rec_strings.append(t_str)
            surrogate_rec_strings.append(s_str)

        # Store results
        results.append(
            {
                "example": i + 1,
                "sequence": sequence,
                "sequence_str": sequence_str,
                "target_recs": target_rec_strings,
                "surrogate_recs": surrogate_rec_strings,
                "similarity": similarity,
            }
        )

    # Print results
    for result in results:
        print(f"\nExample {result['example']}:")
        print(f"Sequence: {result['sequence_str']}")
        print("\nRecommendations:")

        headers = ["Rank", "Target Model", "Surrogate Model"]
        table_data = []

        for j in range(top_k):
            table_data.append(
                [j + 1, result["target_recs"][j], result["surrogate_recs"][j]]
            )

        print(tabulate(table_data, headers=headers))

        similarity = result["similarity"]
        print(
            f"\nSimilarity: {similarity['overlap']}/{top_k} matches ({similarity['overlap_percentage']:.1f}%)"
        )
        print(f"Kendall's tau: {similarity['kendall_tau']:.3f}")
        print("=" * 80)

    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write("MODEL COMPARISON EXAMPLES\n")
            f.write("========================\n\n")

            for result in results:
                f.write(f"Example {result['example']}:\n")
                f.write(f"Sequence: {result['sequence_str']}\n\n")
                f.write("Recommendations:\n")

                for j in range(top_k):
                    f.write(f"  {result['target_recs'][j]}\n")
                    f.write(f"  {result['surrogate_recs'][j]}\n\n")

                similarity = result["similarity"]
                f.write(
                    f"Similarity: {similarity['overlap']}/{top_k} matches ({similarity['overlap_percentage']:.1f}%)\n"
                )
                f.write(f"Kendall's tau: {similarity['kendall_tau']:.3f}\n\n")
                f.write("=" * 80 + "\n\n")

            print(f"Results saved to {output_file}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare model recommendations")
    parser.add_argument(
        "--target-model",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to target model",
    )
    parser.add_argument(
        "--surrogate-model",
        type=str,
        default="checkpoints/surrogate_model.pt",
        help="Path to surrogate model",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/ml-1m", help="Path to MovieLens dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="attack_results/recommendation_comparison.txt",
        help="Output file path",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of recommendations"
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    ratings_file = os.path.join(args.data_path, "ratings.dat")
    movies_file = os.path.join(args.data_path, "movies.dat")

    data = load_movielens(ratings_file)
    num_items = data["num_items"]

    # Load movie data
    movies_data = load_movie_data(movies_file)
    print(f"Loaded data for {len(movies_data)} movies")

    # Load models
    print(f"Loading target model from {args.target_model}")
    target_model = load_model(args.target_model, num_items, args.embedding_dim, device)

    print(f"Loading surrogate model from {args.surrogate_model}")
    surrogate_model = load_model(
        args.surrogate_model, num_items, args.embedding_dim, device
    )

    # Create example sequences
    sequences = create_example_sequences(num_items)

    # Compare examples
    compare_examples(
        target_model,
        surrogate_model,
        sequences,
        movies_data,
        output_file=args.output,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
