import torch
import os

from src.data.data_utils import load_movielens
from src.models.base_model import SimpleSequentialRecommender


# Function to load movie titles from movies.dat
def load_movie_titles(file_path):
    movie_titles = {}
    with open(file_path, "r", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.strip().split("::")
            movie_id = int(parts[0])
            title = parts[1]
            movie_titles[movie_id] = title
    return movie_titles


# Load model
checkpoint_path = "checkpoints/best_model.pt"
data_path = "data/ml-1m.csv"
movies_path = "data/ml-1m/movies.dat"

# Load data to get num_items
data = load_movielens(data_path)

# Load movie titles
movie_titles = load_movie_titles(movies_path)

# Initialize model
model = SimpleSequentialRecommender(num_items=data["num_items"])
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Sample sequence
sequence = [1352, 3896, 3863, 3535, 3798, 3826]  # Example movie IDs

# Get recommendations
with torch.no_grad():
    scores = model.predict_next_item(sequence)
    top_indices = torch.topk(scores, 10).indices.tolist()

# Map sequence and recommendations to movie titles
sequence_titles = [f"{id} ({movie_titles.get(id, 'Unknown')})" for id in sequence]
recommendation_titles = [
    f"{id} ({movie_titles.get(id, 'Unknown')})" for id in top_indices
]

print(f"For sequence:")
for movie in sequence_titles:
    print(f"  - {movie}")

print(f"\nTop recommendations:")
for movie in recommendation_titles:
    print(f"  - {movie}")
