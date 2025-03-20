import os
import torch
import argparse
import numpy as np
from collections import defaultdict

from src.models.base_model import SimpleSequentialRecommender


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Interactive movie recommender")
    parser.add_argument(
        "--model",
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
    title_to_id = {}

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

                full_title = f"{title} ({year})"
                genres = parts[2].split("|")
                movie_titles[movie_id] = (full_title, genres)
                title_to_id[full_title.lower()] = movie_id
    except Exception as e:
        print(f"Error loading movie titles: {e}")

    return movie_titles, title_to_id


def load_model(model_path, num_items, embedding_dim):
    """Load recommendation model"""
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SimpleSequentialRecommender(num_items, embedding_dim)

    # Load state dict
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None

    try:
        # Try loading as a full checkpoint first
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Model has {embedding_dim} embedding dimensions")
        else:
            # Otherwise assume it's just the state dict
            model.load_state_dict(checkpoint)
            print("Loaded model state dict")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model = model.to(device)
    model.eval()
    return model


def search_movies(query, movie_titles, title_to_id):
    """Search for movies matching a query"""
    query = query.lower()
    results = []

    # First try exact match on movie ID
    try:
        movie_id = int(query)
        if movie_id in movie_titles:
            title, genres = movie_titles[movie_id]
            results.append((movie_id, title, genres))
            return results
    except ValueError:
        pass

    # Then try matching on title
    for movie_id, (title, genres) in movie_titles.items():
        if query in title.lower():
            results.append((movie_id, title, genres))
            if len(results) >= 10:  # Limit to 10 results
                break

    return results


def get_model_recommendations(
    model, sequence, top_k=10, movie_titles=None, use_genre_bias=False
):
    """Get model-based recommendations for a sequence"""
    # Check if sequence is empty
    if not sequence:
        return []

    device = next(model.parameters()).device

    if use_genre_bias and movie_titles:
        # Create item_to_genre mapping for genre-biased recommendations
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
        with torch.no_grad():
            recommendations = model.recommend_with_genre_bias(
                sequence, k=top_k, genre_weights=genre_weights
            )

        return recommendations
    else:
        # Standard recommendation method
        # Convert sequence to tensor
        input_ids = torch.tensor([sequence], dtype=torch.long).to(device)

        # Get model predictions
        with torch.no_grad():
            logits = model(input_ids)[0]  # Get logits for first (only) batch item

        # Remove items that are already in the sequence (don't recommend what's already watched)
        for idx in sequence:
            if idx < logits.size(0):
                logits[idx] = float("-inf")

        # Get top-k recommendations
        scores, indices = torch.topk(logits, k=top_k)

        # Convert to list of (id, score) tuples
        recommendations = [
            (idx.item(), score.item()) for idx, score in zip(indices, scores)
        ]

        return recommendations


def get_popular_movies(movie_titles, n=10):
    """Get a list of popular movies (hardcoded for simplicity)"""
    # These are some of the most popular movies in the MovieLens dataset
    popular_ids = [2571, 2858, 296, 1196, 589, 2762, 593, 1198, 318, 356]
    return [
        (movie_id, movie_titles[movie_id][0], movie_titles[movie_id][1])
        for movie_id in popular_ids
        if movie_id in movie_titles
    ][:n]


def main(args=None):
    # Parse arguments
    if args is None:
        args = parse_args()

    # Load movie data
    movie_titles, title_to_id = load_movie_titles()
    print(
        f"Loaded {len(movie_titles)} movie titles with {len(set(g for _, genres in movie_titles.values() for g in genres))} genres"
    )

    # Determine number of items
    num_items = max(movie_titles.keys()) + 1
    print(f"Total number of items: {num_items}")

    # Load model
    model = load_model(args.model, num_items, args.embedding_dim)
    if model is None:
        print("Could not load model, exiting")
        return

    # User's current sequence
    user_sequence = []

    # Print welcome message
    print("\nWelcome to the Interactive Movie Recommender!")
    print("----------------------------------------------")
    print("Commands:")
    print("  search <query>: Search for movies")
    print("  add <id>: Add movie to your sequence")
    print("  remove <id or index>: Remove movie from your sequence")
    print("  clear: Clear your sequence")
    print("  recommend: Get recommendations")
    print("  show: Show your current sequence")
    print("  popular: Show popular movies")
    print("  exit: Exit the program")

    # If using genre bias, show that in welcome message
    if args.use_genre_bias:
        print("\nUsing genre-biased recommendations")

    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            if not user_input:
                continue

            # Split command and arguments
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            # Process command
            if command == "exit":
                print("Goodbye!")
                break

            elif command == "search":
                if not arg:
                    print("Please provide a search query")
                    continue

                print(f"\nSearching for '{arg}':")
                results = search_movies(arg, movie_titles, title_to_id)
                if results:
                    for i, (movie_id, title, genres) in enumerate(results, 1):
                        print(
                            f"  {i}. {title} (ID: {movie_id}) - Genres: {', '.join(genres)}"
                        )
                else:
                    print("No results found")

            elif command == "add":
                try:
                    movie_id = int(arg)
                    if movie_id in movie_titles:
                        user_sequence.append(movie_id)
                        title, genres = movie_titles[movie_id]
                        print(
                            f"Added: {title} (ID: {movie_id}) - Genres: {', '.join(genres)}"
                        )
                    else:
                        print(f"Movie ID {movie_id} not found")
                except ValueError:
                    print("Please provide a valid movie ID")

            elif command == "remove":
                if not user_sequence:
                    print("Your sequence is empty")
                    continue

                try:
                    idx = int(arg)
                    # Check if it's an index or a movie ID
                    if 1 <= idx <= len(user_sequence):  # It's an index
                        movie_id = user_sequence.pop(idx - 1)
                        title, genres = movie_titles.get(
                            movie_id, (f"Movie {movie_id}", [])
                        )
                        print(
                            f"Removed: {title} (ID: {movie_id}) - Genres: {', '.join(genres)}"
                        )
                    elif idx in user_sequence:  # It's a movie ID
                        user_sequence.remove(idx)
                        title, genres = movie_titles.get(idx, (f"Movie {idx}", []))
                        print(
                            f"Removed: {title} (ID: {idx}) - Genres: {', '.join(genres)}"
                        )
                    else:
                        print(f"Index/ID {idx} not found in your sequence")
                except ValueError:
                    print("Please provide a valid index or movie ID")

            elif command == "clear":
                user_sequence = []
                print("Sequence cleared")

            elif command == "show":
                if not user_sequence:
                    print("Your sequence is empty")
                    continue

                print("\nYour current sequence:")
                for i, movie_id in enumerate(user_sequence, 1):
                    title, genres = movie_titles.get(
                        movie_id, (f"Movie {movie_id}", [])
                    )
                    print(
                        f"  {i}. {title} (ID: {movie_id}) - Genres: {', '.join(genres)}"
                    )

            elif command == "recommend":
                if not user_sequence:
                    print(
                        "Your sequence is empty. Add some movies first with 'add <id>'"
                    )
                    continue

                print("\nGenerating recommendations based on your sequence:")
                print("Current sequence:")
                for i, movie_id in enumerate(user_sequence, 1):
                    title, genres = movie_titles.get(
                        movie_id, (f"Movie {movie_id}", [])
                    )
                    print(
                        f"  {i}. {title} (ID: {movie_id}) - Genres: {', '.join(genres)}"
                    )

                # Get model recommendations
                recommendations = get_model_recommendations(
                    model,
                    user_sequence,
                    args.top_k,
                    movie_titles if args.use_genre_bias else None,
                    args.use_genre_bias,
                )

                if recommendations:
                    print("\nRecommended movies:")
                    for i, (movie_id, score) in enumerate(recommendations, 1):
                        title, genres = movie_titles.get(
                            movie_id, (f"Movie {movie_id}", [])
                        )
                        print(
                            f"  {i}. {title} (ID: {movie_id}) - Score: {score:.4f} - Genres: {', '.join(genres)}"
                        )
                else:
                    print("No recommendations available")

            elif command == "popular":
                print("\nPopular movies:")
                for i, (movie_id, title, genres) in enumerate(
                    get_popular_movies(movie_titles), 1
                ):
                    print(
                        f"  {i}. {title} (ID: {movie_id}) - Genres: {', '.join(genres)}"
                    )

            else:
                print(f"Unknown command: {command}")
                print(
                    "Available commands: search, add, remove, clear, recommend, show, popular, exit"
                )

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
