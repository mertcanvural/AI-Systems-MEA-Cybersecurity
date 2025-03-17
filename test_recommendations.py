import os
from collections import Counter
from src.models.base_model import SimpleSequentialRecommender


def load_movie_titles(movies_path):
    """Load movie titles and genres from movies.dat file"""
    movie_titles = {}
    genre_to_movies = {}

    with open(movies_path, "r", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.strip().split("::")
            movie_id = int(parts[0])
            title = parts[1]
            genres = parts[2].split("|")

            # Store movie info
            movie_titles[movie_id] = {"title": title, "genres": genres}

            # Build genre index
            for genre in genres:
                if genre not in genre_to_movies:
                    genre_to_movies[genre] = []
                genre_to_movies[genre].append(movie_id)

    return movie_titles, genre_to_movies


def get_genre_based_recommendations(sequence, movie_titles, genre_to_movies, top_k=10):
    """Get recommendations based on genre matching"""
    # Count genres in the input sequence
    genre_counts = Counter()
    for movie_id in sequence:
        if movie_id in movie_titles:
            genres = movie_titles[movie_id]["genres"]
            genre_counts.update(genres)

    # Get most common genres
    top_genres = [g for g, _ in genre_counts.most_common(3)]

    # Get movies matching these genres
    matching_movies = set()
    for genre in top_genres:
        matching_movies.update(genre_to_movies.get(genre, []))

    # Remove movies that are already in the sequence
    matching_movies = matching_movies - set(sequence)

    # Create scores based on genre match count
    movie_scores = []
    for movie_id in matching_movies:
        genre_match_count = sum(
            1 for g in movie_titles[movie_id]["genres"] if g in top_genres
        )
        movie_scores.append((movie_id, genre_match_count))

    # Sort by score
    movie_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top k
    return movie_scores[:top_k]


def main():
    # Load movie data
    movies_path = "data/ml-1m/movies.dat"
    movie_titles, genre_to_movies = load_movie_titles(movies_path)
    print(f"Loaded {len(movie_titles)} movie titles with {len(genre_to_movies)} genres")

    # Test sequences
    test_sequences = [
        # Children's movies
        [3438, 3439, 3440],  # TMNT movies
        # Input from user question
        [3438, 3439, 2399, 2161, 2162],  # TMNT + NeverEnding Story + Santa Claus
        # Comedy
        [1, 3, 7],  # Toy Story, Grumpier Old Men, Sabrina
        # Action/Sci-Fi
        [1240, 589, 1196],  # Terminator, Terminator 2, Empire Strikes Back
    ]

    for i, sequence in enumerate(test_sequences, 1):
        print(f"\nSequence {i}:")
        # Print the movies in the sequence
        for j, movie_id in enumerate(sequence, 1):
            movie_info = movie_titles.get(movie_id, {})
            title = movie_info.get("title", f"Movie {movie_id}")
            genres = movie_info.get("genres", [])
            genres_str = ", ".join(genres) if genres else "Unknown"
            print(f"  {j}. {title} (ID: {movie_id}) - Genres: {genres_str}")

        # Get recommendations
        top_movies = get_genre_based_recommendations(
            sequence, movie_titles, genre_to_movies, 5
        )

        # Print recommendations
        print("\n  Top 5 recommendations (genre-based):")
        for j, (movie_id, score) in enumerate(top_movies, 1):
            movie_info = movie_titles.get(movie_id, {})
            title = movie_info.get("title", f"Movie {movie_id}")
            genres = movie_info.get("genres", [])
            genres_str = ", ".join(genres) if genres else "Unknown"
            print(
                f"    {j}. {title} (ID: {movie_id}) - Score: {score} - Genres: {genres_str}"
            )


if __name__ == "__main__":
    main()
