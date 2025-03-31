import torch
import random
from src.models.base_model import SimpleSequentialRecommender
from src.data.data_utils import load_movielens


def load_model(model_path, num_items, embedding_dim=256, device=None):
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
        print(f"Error loading model from {model_path}: {e}")
        return None


def load_movie_data():
    """Load movie titles and genres"""
    movies_path = "data/ml-1m/movies.dat"
    movies = {}

    with open(movies_path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            movie_id, title, genres = line.strip().split("::")
            movie_id = int(movie_id)
            movies[movie_id] = {"title": title, "genres": genres}

    return movies


def get_model_recommendations(model, sequence, top_k=10, device=None):
    """Get top-k recommendations from model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare sequence for model
    seq_tensor = torch.LongTensor([sequence]).to(device)

    # Get predictions
    with torch.no_grad():
        logits = model(seq_tensor)

    # Get top-k recommendations
    _, indices = torch.topk(logits, k=top_k, dim=1)

    # Convert to list
    return indices[0].cpu().numpy().tolist()


def format_recommendation_list(movie_ids, movies_data, header):
    """Format list of movie recommendations with title and genres"""
    result = [f"\n{header}\n" + "=" * 80]

    for i, movie_id in enumerate(movie_ids):
        if movie_id in movies_data:
            movie = movies_data[movie_id]
            title = movie["title"]
            genres = movie["genres"]
            result.append(f"{i+1}. [{movie_id}] {title} - {genres}")
        else:
            result.append(f"{i+1}. [ID: {movie_id}] Unknown Movie")

    return "\n".join(result)


def compare_recommendations(
    user_sequences, movies_data, models_info, num_samples=5, seed=42
):
    """Compare recommendations from different models for the same users"""
    random.seed(seed)

    # Get random user sequences
    user_ids = list(user_sequences.keys())
    selected_users = random.sample(user_ids, min(num_samples, len(user_ids)))

    results = []

    for user_id in selected_users:
        sequence = user_sequences[user_id]

        # Skip if sequence is too short
        if len(sequence) < 3:
            continue

        div_line = "=" * 80
        user_result = [f"\n\n{div_line}"]
        user_result.append(f"USER {user_id} MOVIE RECOMMENDATIONS")
        user_result.append(f"{div_line}")

        # Show user's history (last few movies they watched)
        history = sequence[-10:] if len(sequence) > 10 else sequence
        user_result.append("\nMOVIE HISTORY:")
        user_result.append("-" * 40)

        for i, movie_id in enumerate(history):
            movie_info = movies_data.get(movie_id, {})
            title = movie_info.get("title", "Unknown")
            genres = movie_info.get("genres", "Unknown")
            user_result.append(f"{i+1}. [{movie_id}] {title} - {genres}")

        # Get recommendations from original model (ground truth)
        original_model = models_info.get("Original Target")
        if original_model:
            original_recs = get_model_recommendations(original_model, sequence)
            user_result.append("\nORIGINAL TARGET MODEL RECOMMENDATIONS:")
            user_result.append("-" * 40)
            for i, movie_id in enumerate(original_recs):
                movie_info = movies_data.get(movie_id, {})
                title = movie_info.get("title", "Unknown")
                genres = movie_info.get("genres", "Unknown")
                user_result.append(f"{i+1}. [{movie_id}] {title} - {genres}")

        # Get recommendations from original surrogate (undefended attack)
        original_surrogate = models_info.get("Original Surrogate (Attacker)")
        if original_surrogate:
            surrogate_recs = get_model_recommendations(original_surrogate, sequence)
            user_result.append("\nUNDEFENDED ATTACKER MODEL RECOMMENDATIONS:")
            user_result.append("-" * 40)
            for i, movie_id in enumerate(surrogate_recs):
                movie_info = movies_data.get(movie_id, {})
                title = movie_info.get("title", "Unknown")
                genres = movie_info.get("genres", "Unknown")
                user_result.append(f"{i+1}. [{movie_id}] {title} - {genres}")

            # Calculate and display overlap with original
            overlap = len(set(original_recs) & set(surrogate_recs))
            overlap_percent = (overlap / len(original_recs)) * 100
            user_result.append(
                f"\nUndefended Attack Overlap: {overlap}/{len(original_recs)} "
                f"movies ({overlap_percent:.1f}%)"
            )

        # Get recommendations from defended surrogate
        defended_surrogate = models_info.get("Defended Surrogate (Attacker)")
        if defended_surrogate:
            defended_recs = get_model_recommendations(defended_surrogate, sequence)
            user_result.append("\nDEFENDED ATTACKER MODEL RECOMMENDATIONS:")
            user_result.append("-" * 40)
            for i, movie_id in enumerate(defended_recs):
                movie_info = movies_data.get(movie_id, {})
                title = movie_info.get("title", "Unknown")
                genres = movie_info.get("genres", "Unknown")
                user_result.append(f"{i+1}. [{movie_id}] {title} - {genres}")

            # Calculate and display overlap with original
            overlap = len(set(original_recs) & set(defended_recs))
            overlap_percent = (overlap / len(original_recs)) * 100
            user_result.append(
                f"\nDefended Attack Overlap: {overlap}/{len(original_recs)} "
                f"movies ({overlap_percent:.1f}%)"
            )

        results.append("\n".join(user_result))

    return "\n".join(results)


def main():
    """Display sample recommendations from all models"""
    print("Loading models and data...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load movie data
    movies_data = load_movie_data()
    print(f"Loaded data for {len(movies_data)} movies")

    # Load user sequences
    data = load_movielens("data/ml-1m/ratings.dat")
    user_sequences = data["user_sequences"]
    num_items = data["num_items"]
    print(f"Loaded sequences for {len(user_sequences)} users")

    # Load models
    models_info = {
        "Original Target": load_model(
            "checkpoints/best_model.pt", num_items, device=device
        ),
        "Original Surrogate (Attacker)": load_model(
            "defense_results/original_surrogate.pt", num_items, device=device
        ),
        "Defended Surrogate (Attacker)": load_model(
            "defense_results/defended_surrogate.pt", num_items, device=device
        ),
    }

    # Check which models were loaded successfully
    for name, model in models_info.items():
        if model is None:
            print(f"Warning: Failed to load {name} model")
        else:
            print(f"Successfully loaded {name} model")

    # Display recommendations comparison
    print("\n\nGenerating recommendation comparisons...")
    recommendations = compare_recommendations(
        user_sequences, movies_data, models_info, num_samples=5
    )

    # Print recommendations
    print("\n\n" + "=" * 100)
    print("RECOMMENDATION MODEL EXTRACTION ATTACK COMPARISON")
    print("=" * 100)
    print(recommendations)

    # Save recommendations to file
    output_file = "defense_results/attack_comparison.txt"
    with open(output_file, "w") as f:
        f.write("RECOMMENDATION MODEL EXTRACTION ATTACK COMPARISON\n")
        f.write("=" * 100 + "\n")
        f.write(recommendations)

    print(f"\nRecommendations saved to {output_file}")


if __name__ == "__main__":
    main()
