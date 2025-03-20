import os
import torch
import argparse
from tabulate import tabulate

from src.models.base_model import SimpleSequentialRecommender


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate genre-biased recommendations"
    )
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
        "--top-k", type=int, default=10, help="Number of recommendations to show"
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
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model.to(device)


def get_standard_recommendations(model, sequence, top_k=10):
    """Get standard model recommendations"""
    device = next(model.parameters()).device

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


def get_genre_biased_recommendations(model, sequence, movie_titles, top_k=10):
    """Get genre-biased recommendations"""
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


def calculate_genre_overlap(sequence, recommendations, movie_titles):
    """Calculate genre overlap between sequence and recommendations"""
    # Get sequence genres
    sequence_genres = set()
    for movie_id in sequence:
        _, genres = movie_titles.get(movie_id, ("", []))
        sequence_genres.update(genres)

    # Get recommendation genres
    rec_genres = set()
    for movie_id, _ in recommendations:
        _, genres = movie_titles.get(movie_id, ("", []))
        rec_genres.update(genres)

    # Calculate overlap using Jaccard similarity
    if not sequence_genres or not rec_genres:
        return 0.0

    intersection = len(sequence_genres.intersection(rec_genres))
    union = len(sequence_genres.union(rec_genres))
    return intersection / union


def evaluate_test_sequences(model, movie_titles, top_k=10):
    """Evaluate and compare recommendation methods on test sequences"""
    # Define test sequences with different genre profiles
    test_sequences = [
        {
            "name": "Children's Movies",
            "ids": [3438, 3439, 2399, 2161, 2162],  # Ninja Turtles, Santa, NeverEnding
            "desc": "A sequence of children's and family movies",
        },
        {
            "name": "Action & Sci-Fi",
            "ids": [1240, 589, 1196, 2916],  # Terminator, T2, Star Wars, Matrix
            "desc": "A sequence of action and sci-fi blockbusters",
        },
        {
            "name": "Romance & Comedy",
            "ids": [1265, 1721, 2128, 2706],  # Groundhog Day, When Harry Met Sally, etc
            "desc": "A sequence of romantic comedies",
        },
        {
            "name": "Mixed Genres",
            "ids": [
                1,
                100,
                590,
                1210,
                1617,
            ],  # Toy Story, Fargo, Jurassic Park, Star Wars
            "desc": "A diverse sequence with varied genres",
        },
    ]

    results = []

    for sequence_info in test_sequences:
        sequence = sequence_info["ids"]
        print(f"\n=== {sequence_info['name']} ===")
        print(f"{sequence_info['desc']}")

        # Print the input sequence
        print("\nInput sequence:")
        for i, movie_id in enumerate(sequence, 1):
            title, genres = movie_titles.get(
                movie_id, (f"Unknown Movie {movie_id}", [])
            )
            print(f"  {i}. {title} (ID: {movie_id}) - Genres: {', '.join(genres)}")

        # Get standard recommendations
        std_recommendations = get_standard_recommendations(model, sequence, top_k=top_k)

        # Get genre-biased recommendations
        biased_recommendations = get_genre_biased_recommendations(
            model, sequence, movie_titles, top_k=top_k
        )

        # Calculate genre overlap
        std_overlap = calculate_genre_overlap(
            sequence, std_recommendations, movie_titles
        )
        biased_overlap = calculate_genre_overlap(
            sequence, biased_recommendations, movie_titles
        )

        # Print standard recommendations
        print("\nStandard Model Recommendations:")
        for i, (movie_id, score) in enumerate(std_recommendations, 1):
            title, genres = movie_titles.get(
                movie_id, (f"Unknown Movie {movie_id}", [])
            )
            print(
                f"  {i}. {title} (ID: {movie_id}) - Score: {score:.4f} - Genres: {', '.join(genres)}"
            )
        print(f"Genre overlap: {std_overlap:.4f}")

        # Print genre-biased recommendations
        print("\nGenre-Biased Recommendations:")
        for i, (movie_id, score) in enumerate(biased_recommendations, 1):
            title, genres = movie_titles.get(
                movie_id, (f"Unknown Movie {movie_id}", [])
            )
            print(
                f"  {i}. {title} (ID: {movie_id}) - Score: {score:.4f} - Genres: {', '.join(genres)}"
            )
        print(f"Genre overlap: {biased_overlap:.4f}")

        # Store results
        results.append(
            {
                "name": sequence_info["name"],
                "standard_overlap": std_overlap,
                "biased_overlap": biased_overlap,
                "improvement": biased_overlap - std_overlap,
                "percent_improvement": (
                    ((biased_overlap / std_overlap) - 1) * 100
                    if std_overlap > 0
                    else float("inf")
                ),
            }
        )

    # Print summary table
    print("\n=== Summary of Genre Overlap ===")
    table_data = []
    for r in results:
        table_data.append(
            [
                r["name"],
                f"{r['standard_overlap']:.4f}",
                f"{r['biased_overlap']:.4f}",
                f"{r['improvement']:.4f}",
                (
                    f"{r['percent_improvement']:.1f}%"
                    if r["percent_improvement"] != float("inf")
                    else "N/A"
                ),
            ]
        )

    headers = ["Sequence", "Standard", "Genre-Biased", "Improvement", "% Improvement"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    # Overall improvement
    avg_std = sum(r["standard_overlap"] for r in results) / len(results)
    avg_biased = sum(r["biased_overlap"] for r in results) / len(results)
    print(
        f"\nAverage genre overlap - Standard: {avg_std:.4f}, Genre-Biased: {avg_biased:.4f}"
    )
    print(
        f"Overall improvement: {avg_biased - avg_std:.4f} ({((avg_biased / avg_std) - 1) * 100:.1f}%)"
    )


def main(args=None):
    """Main function"""
    if args is None:
        args = parse_args()

    # Load movie titles and genres
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

    # Evaluate test sequences
    evaluate_test_sequences(model, movie_titles, args.top_k)


if __name__ == "__main__":
    main()
