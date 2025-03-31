import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict

from src.models.base_model import SimpleSequentialRecommender
from src.data.data_utils import (
    load_movielens,
    create_train_val_test_splits,
    create_data_loaders,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate recommendation model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of recommendations to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--use-genre-bias",
        action="store_true",
        help="Use genre bias for recommendations",
    )
    return parser.parse_args()


def load_movie_titles():
    """Load movie titles and genres"""
    movies_path = "data/ml-1m/movies.dat"
    movie_titles = {}

    try:
        with open(movies_path, "r", encoding="ISO-8859-1") as f:
            for line in f:
                parts = line.strip().split("::")
                movie_id = int(parts[0])
                title_genre = parts[1].split("(")
                title = title_genre[0].strip()

                year = ""
                if len(title_genre) > 1 and title_genre[-1].endswith(")"):
                    year = title_genre[-1].strip(")")

                genres = parts[2].split("|")
                movie_titles[movie_id] = (f"{title} ({year})", genres)
    except Exception as e:
        print(f"Error loading movie titles: {e}")

    return movie_titles


def load_model(model_path, num_items, embedding_dim=64):
    """Load a trained recommendation model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SimpleSequentialRecommender(num_items, embedding_dim)

    # Load state dict
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(
                f"Model metrics - Val loss: {checkpoint.get('val_loss', 'unknown'):.4f}, "
                f"Val Hit@10: {checkpoint.get('val_hit_rate', 'unknown'):.4f}"
            )
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model


def evaluate_on_test_set(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    hits = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            test_loss += loss.item()

            # Hit@10 metric
            _, top_indices = torch.topk(logits, k=10, dim=1)
            for i, label in enumerate(labels):
                if label in top_indices[i]:
                    hits += 1
                total += 1

    avg_loss = test_loss / len(test_loader)
    hit_rate = hits / total if total > 0 else 0

    return avg_loss, hit_rate


def get_recommendations(model, sequence, top_k=10, device=None):
    """Get recommendations for a sequence"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        # Convert sequence to tensor
        input_ids = torch.tensor([sequence], dtype=torch.long).to(device)

        # Forward pass
        logits = model(input_ids)

        # Get top-k recommendations
        scores, indices = torch.topk(logits, k=top_k, dim=1)

        # Convert to list
        recommendations = [
            (idx.item(), score.item()) for idx, score in zip(indices[0], scores[0])
        ]

    return recommendations


def get_genre_biased_recommendations(
    model, sequence, movie_titles, top_k=10, device=None
):
    """Get genre-biased recommendations for a sequence"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create item_to_genre mapping
    item_to_genre = {}
    for movie_id, (_, genres) in movie_titles.items():
        item_to_genre[movie_id] = genres

    # Set genre mapping in model
    model.set_genre_mapping(item_to_genre)

    # Get sequence genres to create weights
    sequence_genres = set()
    for movie_id in sequence:
        _, genres = movie_titles.get(movie_id, ("", []))
        sequence_genres.update(genres)

    # Create genre weights - boost genres in the sequence
    genre_weights = {genre: 0.5 for genre in sequence_genres}

    # Get recommendations with genre bias
    model.eval()
    with torch.no_grad():
        recommendations = model.recommend_with_genre_bias(
            sequence, k=top_k, genre_weights=genre_weights
        )

    return recommendations


def evaluate_sample_sequences(
    model, movie_titles, device, top_k=10, use_genre_bias=False
):
    """Evaluate model on sample sequences"""
    # Define test sequences
    test_sequences = [
        # Children's movies sequence
        [3438, 3439, 3440],  # Teenage Mutant Ninja Turtles trilogy
        # Mixed genre sequence
        [1, 3, 7],  # Toy Story, Grumpier Old Men, Sabrina
        # Action/sci-fi sequence
        [1240, 589, 1196],  # Terminator, T2, Star Wars Empire Strikes Back
        # Longer children's sequence
        [3438, 3439, 2399, 2161, 2162],  # TMNT + Santa Claus + NeverEnding Story
    ]

    results = []

    for i, sequence in enumerate(test_sequences, 1):
        print(f"\n=== Sequence {i} ===")

        # Print the input sequence
        print("Input sequence:")
        for j, movie_id in enumerate(sequence, 1):
            title, genres = movie_titles.get(
                movie_id, (f"Unknown Movie {movie_id}", [])
            )
            print(f"  {j}. {title} (ID: {movie_id}) - Genres: {', '.join(genres)}")

        # Get recommendations based on method
        if use_genre_bias:
            print("\nUsing genre-biased recommendations")
            recommendations = get_genre_biased_recommendations(
                model, sequence, movie_titles, top_k=top_k, device=device
            )
        else:
            print("\nUsing standard model recommendations")
            recommendations = get_recommendations(
                model, sequence, top_k=top_k, device=device
            )

        # Print recommendations
        print("\nRecommendations:")
        for j, (movie_id, score) in enumerate(recommendations, 1):
            title, genres = movie_titles.get(
                movie_id, (f"Unknown Movie {movie_id}", [])
            )
            print(
                f"  {j}. {title} (ID: {movie_id}) - Score: {score:.4f} - Genres: {', '.join(genres)}"
            )

        # Store results for analysis
        sequence_genres = set()
        for movie_id in sequence:
            _, genres = movie_titles.get(movie_id, ("", []))
            sequence_genres.update(genres)

        rec_genres = set()
        for movie_id, _ in recommendations:
            _, genres = movie_titles.get(movie_id, ("", []))
            rec_genres.update(genres)

        # Calculate genre overlap
        genre_overlap = (
            len(sequence_genres.intersection(rec_genres))
            / len(sequence_genres.union(rec_genres))
            if sequence_genres and rec_genres
            else 0
        )

        results.append(
            {
                "sequence_id": i,
                "sequence_len": len(sequence),
                "genre_overlap": genre_overlap,
            }
        )

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    avg_genre_overlap = sum(r["genre_overlap"] for r in results) / len(results)
    print(f"Average genre overlap: {avg_genre_overlap:.4f}")

    # Create table
    table_data = []
    for r in results:
        table_data.append(
            [
                r["sequence_id"],
                r["sequence_len"],
                f"{r['genre_overlap']:.4f}",
            ]
        )

    headers = ["Sequence", "Length", "Genre Overlap"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))


def evaluate_recommendation_methods(model, movie_titles, device, top_k=10):
    """Compare standard and genre-biased recommendations"""
    print("\n=== Standard Recommendations ===")
    standard_results = evaluate_sample_sequences(
        model, movie_titles, device, top_k, use_genre_bias=False
    )

    print("\n=== Genre-Biased Recommendations ===")
    biased_results = evaluate_sample_sequences(
        model, movie_titles, device, top_k, use_genre_bias=True
    )


def main(args=None):
    """Main function"""
    if args is None:
        args = parse_args()

    # Load movie titles
    movie_titles = load_movie_titles()
    print(f"Loaded {len(movie_titles)} movie titles")

    # Get number of items
    num_items = max(movie_titles.keys()) + 1
    print(f"Number of items: {num_items}")

    # Load model
    model = load_model(args.model_path, num_items, args.embedding_dim)
    if model is None:
        print("Could not load model, exiting")
        return

    model = model.to(args.device)
    model.eval()

    # Compare recommendation methods if requested
    if args.use_genre_bias:
        evaluate_recommendation_methods(model, movie_titles, args.device, args.top_k)
    else:
        # Evaluate on sample sequences
        evaluate_sample_sequences(model, movie_titles, args.device, args.top_k)

    # Optionally evaluate on test set
    test_results = None
    if os.path.exists("data/ml-1m/ratings.dat"):
        print("\n=== Evaluating on test set ===")
        data = load_movielens("data/ml-1m/ratings.dat", min_rating=4.0)
        splits = create_train_val_test_splits(data["user_sequences"])
        loaders = create_data_loaders(splits, batch_size=128)

        test_loss, test_hit_rate = evaluate_on_test_set(
            model, loaders["test"], args.device
        )
        print(f"Test loss: {test_loss:.4f}, Test Hit@10: {test_hit_rate:.4f}")
        test_results = {"test_loss": test_loss, "test_hit_rate": test_hit_rate}

    return {"model": model, "test_results": test_results}


if __name__ == "__main__":
    main()
