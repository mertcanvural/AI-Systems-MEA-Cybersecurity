import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse

from src.models.base_model import SimpleSequentialRecommender
from src.data.data_utils import (
    load_movielens,
    create_train_val_test_splits,
    create_data_loaders,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize model performance")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="checkpoints/metrics.json",
        help="Path to training metrics JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension"
    )
    return parser.parse_args()


def load_model(model_path, num_items, embedding_dim=256):
    """Load a trained recommendation model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SimpleSequentialRecommender(num_items, embedding_dim)

    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Model val loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
            print(f"Model val Hit@10: {checkpoint.get('val_hit_rate', 'unknown'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model.to(device)


def load_movie_data():
    """Load movie data and genres"""
    movies_path = "data/ml-1m/movies.dat"
    movies = {}
    genre_to_idx = {}
    all_genres = set()

    with open(movies_path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.strip().split("::")
            movie_id = int(parts[0])
            title = parts[1]
            genres = parts[2].split("|")

            movies[movie_id] = {"title": title, "genres": genres}
            all_genres.update(genres)

    # Create genre mapping
    for idx, genre in enumerate(sorted(all_genres)):
        genre_to_idx[genre] = idx

    return movies, genre_to_idx, list(all_genres)


def plot_training_metrics(metrics_path, output_dir):
    """Plot training metrics from saved JSON file"""
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_loss"], label="Training Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Plot hit rate
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_hit_rate"], label="Training Hit@10")
    plt.plot(metrics["val_hit_rate"], label="Validation Hit@10")
    plt.xlabel("Epoch")
    plt.ylabel("Hit@10")
    plt.title("Training and Validation Hit@10")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "hit_rate_curve.png"))
    plt.close()

    # Print final metrics
    final_epoch = len(metrics["train_loss"])
    print(f"Final training metrics after {final_epoch} epochs:")
    print(f"  Train loss: {metrics['train_loss'][-1]:.4f}")
    print(f"  Validation loss: {metrics['val_loss'][-1]:.4f}")
    print(f"  Train Hit@10: {metrics['train_hit_rate'][-1]:.4f}")
    print(f"  Validation Hit@10: {metrics['val_hit_rate'][-1]:.4f}")


def evaluate_on_test_set(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    hits = 0
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0.0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            test_loss += loss.item()

            # Get predictions
            _, top_indices = torch.topk(logits, k=10, dim=1)

            for i, label in enumerate(labels):
                # Check if label is in top-k predictions
                found_indices = (top_indices[i] == label).nonzero(as_tuple=True)[0]
                if len(found_indices) > 0:
                    rank = found_indices[0].item() + 1  # +1 for 1-indexing
                    mrr += 1.0 / rank
                    hits += 1

                    # Check different cutoffs
                    if rank == 1:
                        hits_at_1 += 1
                    if rank <= 5:
                        hits_at_5 += 1
                    if rank <= 10:
                        hits_at_10 += 1

                total += 1

    # Calculate metrics
    avg_loss = test_loss / len(test_loader)
    hit_rate_1 = hits_at_1 / total if total > 0 else 0
    hit_rate_5 = hits_at_5 / total if total > 0 else 0
    hit_rate_10 = hits_at_10 / total if total > 0 else 0
    mrr = mrr / total if total > 0 else 0

    return {
        "test_loss": avg_loss,
        "hit_rate_1": hit_rate_1,
        "hit_rate_5": hit_rate_5,
        "hit_rate_10": hit_rate_10,
        "mrr": mrr,
    }


def create_genre_confusion_matrix(
    model, test_sequences, movies, genres, genre_to_idx, output_dir
):
    """Create a confusion matrix for genre predictions"""
    device = next(model.parameters()).device
    genre_true = []
    genre_pred = []

    # Process each test sequence
    for sequence in test_sequences[
        :1000
    ]:  # Limit to 1000 to avoid excessive computation
        if len(sequence) < 2:
            continue

        # Split into input and target
        input_seq = sequence[:-1]
        target = sequence[-1]

        # Get target genres
        if target in movies:
            target_genres = movies[target]["genres"]
        else:
            continue

        # Convert sequence to tensor
        input_ids = torch.tensor([input_seq], dtype=torch.long).to(device)

        # Get model predictions
        with torch.no_grad():
            logits = model(input_ids)

        # Get top prediction
        _, top_indices = torch.topk(logits, k=1, dim=1)
        pred_id = top_indices[0][0].item()

        # Get predicted genres
        if pred_id in movies:
            pred_genres = movies[pred_id]["genres"]
        else:
            continue

        # Record the primary genre for confusion matrix
        if target_genres and pred_genres:
            genre_true.append(genre_to_idx.get(target_genres[0], -1))
            genre_pred.append(genre_to_idx.get(pred_genres[0], -1))

    # Only keep valid genre indices
    valid_indices = [
        (t, p) for t, p in zip(genre_true, genre_pred) if t >= 0 and p >= 0
    ]
    genre_true = [pair[0] for pair in valid_indices]
    genre_pred = [pair[1] for pair in valid_indices]

    # Create confusion matrix
    cm = confusion_matrix(genre_true, genre_pred, labels=range(len(genres)))

    # Normalize the confusion matrix
    # Avoid division by zero by adding a small epsilon to rows with zero sum
    row_sums = cm.sum(axis=1)
    # Add a small epsilon to avoid division by zero
    row_sums_safe = np.array([max(sum, 1e-10) for sum in row_sums])
    cm_norm = cm.astype("float") / row_sums_safe[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace any remaining NaN with 0

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=genres,
        yticklabels=genres,
    )
    plt.title("Genre Prediction Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Genre")
    plt.ylabel("True Genre")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "genre_confusion_matrix.png"))
    plt.close()

    return cm, cm_norm


def analyze_genre_performance(model, test_sequences, movies, genres, output_dir):
    """Analyze model performance by genre"""
    device = next(model.parameters()).device
    genre_metrics = {genre: {"count": 0, "hits": 0} for genre in genres}

    # Process each test sequence
    for sequence in test_sequences[:1000]:  # Limit to 1000 sequences
        if len(sequence) < 2:
            continue

        # Split into input and target
        input_seq = sequence[:-1]
        target = sequence[-1]

        # Get target genres
        if target in movies:
            target_genres = movies[target]["genres"]
        else:
            continue

        # Convert sequence to tensor
        input_ids = torch.tensor([input_seq], dtype=torch.long).to(device)

        # Get model predictions
        with torch.no_grad():
            logits = model(input_ids)

        # Get top-10 predictions
        _, top_indices = torch.topk(logits, k=10, dim=1)
        preds = top_indices[0].tolist()

        # Check if hit
        hit = target in preds

        # Update metrics for each genre
        for genre in target_genres:
            if genre in genre_metrics:
                genre_metrics[genre]["count"] += 1
                if hit:
                    genre_metrics[genre]["hits"] += 1

    # Calculate hit rate per genre
    hit_rates = {}
    for genre, metrics in genre_metrics.items():
        if metrics["count"] > 0:
            hit_rates[genre] = metrics["hits"] / metrics["count"]
        else:
            hit_rates[genre] = 0

    # Plot genre performance
    genres_sorted = sorted(hit_rates.keys(), key=lambda x: hit_rates[x], reverse=True)
    hit_rates_sorted = [hit_rates[g] for g in genres_sorted]
    counts_sorted = [genre_metrics[g]["count"] for g in genres_sorted]

    # Filter out genres with too few samples
    min_samples = 10
    filtered_indices = [
        i for i, count in enumerate(counts_sorted) if count >= min_samples
    ]
    genres_filtered = [genres_sorted[i] for i in filtered_indices]
    hit_rates_filtered = [hit_rates_sorted[i] for i in filtered_indices]
    counts_filtered = [counts_sorted[i] for i in filtered_indices]

    # Plot hit rate by genre
    plt.figure(figsize=(12, 8))
    bars = plt.bar(genres_filtered, hit_rates_filtered)
    plt.xlabel("Genre")
    plt.ylabel("Hit@10 Rate")
    plt.title("Hit@10 Rate by Genre")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add sample count as text on each bar
    for i, (bar, count) in enumerate(zip(bars, counts_filtered)):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            0.03,
            f"n={count}",
            ha="center",
            va="bottom",
            rotation=90,
            color="black",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hit_rate_by_genre.png"))
    plt.close()

    # Print top and bottom 5 genres
    print("\nTop 5 genres by Hit@10 rate:")
    for genre, hit_rate, count in zip(
        genres_filtered[:5], hit_rates_filtered[:5], counts_filtered[:5]
    ):
        print(f"  {genre}: {hit_rate:.4f} (n={count})")

    print("\nBottom 5 genres by Hit@10 rate:")
    for genre, hit_rate, count in zip(
        genres_filtered[-5:], hit_rates_filtered[-5:], counts_filtered[-5:]
    ):
        print(f"  {genre}: {hit_rate:.4f} (n={count})")

    return genre_metrics, hit_rates


def create_summary_figure(test_metrics, hit_rates, output_dir):
    """Create a single comprehensive performance summary figure"""
    plt.figure(figsize=(12, 10))

    # Create a 2x2 grid for subplots
    plt.subplot(2, 2, 1)
    # 1. Accuracy metrics bar chart
    accuracy_metrics = [
        test_metrics["hit_rate_1"],
        test_metrics["hit_rate_5"],
        test_metrics["hit_rate_10"],
    ]
    plt.bar(["Hit@1", "Hit@5", "Hit@10"], accuracy_metrics, color="royalblue")
    plt.ylabel("Accuracy")
    plt.title("Recommendation Accuracy")
    # Add percentage labels on each bar
    for i, v in enumerate(accuracy_metrics):
        plt.text(i, v + 0.002, f"{v*100:.2f}%", ha="center")
    plt.ylim(0, max(accuracy_metrics) * 1.2)  # Leave space for labels

    # 2. Top performing genres
    plt.subplot(2, 2, 2)
    genres_sorted = sorted(hit_rates.keys(), key=lambda x: hit_rates[x], reverse=True)
    top_genres = genres_sorted[:5]
    top_rates = [hit_rates[g] for g in top_genres]
    plt.bar(top_genres, top_rates, color="forestgreen")
    plt.ylabel("Hit@10 Rate")
    plt.title("Top 5 Genres by Performance")
    plt.xticks(rotation=30, ha="right")
    # Add percentage labels
    for i, v in enumerate(top_rates):
        plt.text(i, v + 0.005, f"{v*100:.1f}%", ha="center")
    plt.ylim(0, max(top_rates) * 1.2)  # Leave space for labels

    # 3. MRR and position information
    plt.subplot(2, 2, 3)
    mrr = test_metrics["mrr"]
    avg_position = 1 / mrr if mrr > 0 else float("inf")
    plt.bar(
        ["MRR", "Avg. Position"], [mrr, min(avg_position / 100, 0.5)], color="orange"
    )
    plt.ylabel("Value")
    plt.title("Ranking Performance")
    # Add labels with the actual values
    plt.text(0, mrr + 0.002, f"{mrr:.4f}", ha="center")
    plt.text(
        1, min(avg_position / 100, 0.5) + 0.002, f"{avg_position:.1f}", ha="center"
    )

    # 4. Model summary box
    plt.subplot(2, 2, 4)
    plt.axis("off")
    summary_text = (
        f"MODEL PERFORMANCE SUMMARY\n"
        f"-------------------------\n\n"
        f"Test Loss: {test_metrics['test_loss']:.4f}\n\n"
        f"Hit@10: {test_metrics['hit_rate_10']*100:.2f}%\n"
        f"MRR: {test_metrics['mrr']:.4f}\n"
        f"Avg Position: {avg_position:.1f}\n\n"
        f"Best Genre: {top_genres[0]} ({top_rates[0]*100:.1f}%)\n"
        f"Worst Genre: {genres_sorted[-1]} ({hit_rates[genres_sorted[-1]]*100:.1f}%)\n\n"
        f"Performance vs Random: {test_metrics['hit_rate_10']/0.0025:.1f}x better"
    )
    plt.text(
        0.5,
        0.5,
        summary_text,
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=1", facecolor="lightgrey", alpha=0.5),
    )

    plt.tight_layout()
    summary_path = os.path.join(output_dir, "model_performance_summary.png")
    plt.savefig(summary_path)
    plt.close()
    print(f"Performance summary saved to {summary_path}")

    return summary_path


def main(args=None):
    """Main function"""
    if args is None:
        args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot training metrics if available
    if os.path.exists(args.metrics_path):
        print(f"Plotting training metrics from {args.metrics_path}")
        plot_training_metrics(args.metrics_path, args.output_dir)
    else:
        print(f"Metrics file not found: {args.metrics_path}")

    # Load movie data
    print("Loading movie data...")
    movies, genre_to_idx, all_genres = load_movie_data()
    print(f"Loaded data for {len(movies)} movies with {len(all_genres)} genres")

    # Load dataset
    print("Loading MovieLens dataset...")
    data = load_movielens("data/ml-1m/ratings.dat", min_rating=4.0)
    user_sequences = data["user_sequences"]
    num_items = data["num_items"]
    print(f"Loaded {len(user_sequences)} user sequences with {num_items} unique items")

    # Create dataset splits
    print("Creating dataset splits...")
    splits = create_train_val_test_splits(user_sequences)
    loaders = create_data_loaders(splits, batch_size=128)

    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, num_items, args.embedding_dim)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Set device
    device = next(model.parameters()).device
    print(f"Using device: {device}")

    # Evaluate on test set
    print("Evaluating model on test set...")
    test_metrics = evaluate_on_test_set(model, loaders["test"], device)
    print("\nTest set metrics:")
    print(f"  Loss: {test_metrics['test_loss']:.4f}")
    print(f"  Hit@1: {test_metrics['hit_rate_1']:.4f}")
    print(f"  Hit@5: {test_metrics['hit_rate_5']:.4f}")
    print(f"  Hit@10: {test_metrics['hit_rate_10']:.4f}")
    print(f"  MRR: {test_metrics['mrr']:.4f}")

    # Create confusion matrix
    print("Creating genre confusion matrix...")
    test_sequences = []
    for i in range(len(loaders["test"].dataset)):
        batch = loaders["test"].dataset[i]
        sequence = batch["input_ids"].tolist()
        # Remove padding (zeros)
        sequence = [s for s in sequence if s > 0]
        test_sequences.append(sequence)

    cm, cm_norm = create_genre_confusion_matrix(
        model, test_sequences, movies, all_genres, genre_to_idx, args.output_dir
    )

    # Analyze genre performance
    print("Analyzing performance by genre...")
    genre_metrics, hit_rates = analyze_genre_performance(
        model, test_sequences, movies, all_genres, args.output_dir
    )

    # Create comprehensive summary figure
    print("Creating performance summary figure...")
    summary_path = create_summary_figure(test_metrics, hit_rates, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")

    return {
        "test_metrics": test_metrics,
        "genre_metrics": genre_metrics,
        "hit_rates": hit_rates,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    main()
