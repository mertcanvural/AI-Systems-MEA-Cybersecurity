import argparse
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Import your existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from show_recommendations import load_model


class EmbeddingEncoder(torch.nn.Module):
    """Encoder that works with embeddings directly instead of item IDs"""

    def __init__(self, embedding_dim=256, output_dim=128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim),
            torch.nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.mean(x, dim=1)
        # Apply encoder and normalize output for better distance calculations
        encoded = self.encoder(x)
        return torch.nn.functional.normalize(encoded, p=2, dim=1)


def main():
    parser = argparse.ArgumentParser(
        description="Balanced SEAT defense implementation based on the original paper"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--encoder_path", type=str, required=True)
    parser.add_argument("--attack_data", type=str, required=True)
    parser.add_argument("--benign_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="defense_results")
    parser.add_argument(
        "--target_fpr", type=float, default=0.1, help="Target false positive rate"
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("figures/seat", exist_ok=True)

    # Load model and encoder
    model = load_model(args.model_path, 3953, 256, device)
    encoder_data = torch.load(args.encoder_path)
    encoder = EmbeddingEncoder(256, 128)
    encoder.load_state_dict(
        encoder_data
        if not isinstance(encoder_data, dict)
        else encoder_data["state_dict"]
    )
    encoder.to(device)
    encoder.eval()

    # Load queries
    attack_queries = torch.load(args.attack_data)
    benign_dataset = torch.load(args.benign_data)
    benign_queries_list = [
        benign_dataset[i][0] for i in range(min(len(benign_dataset), 1000))
    ]
    benign_queries = torch.stack(benign_queries_list)

    # Use all attack data for better evaluation
    attack_subset = attack_queries
    print(
        f"Using {len(attack_subset)} attack queries and {len(benign_queries)} "
        f"benign queries"
    )

    # SEAT detection approach (following original paper)
    # 1. Encode all queries to get embeddings
    # 2. Compute pairwise distances between all query embeddings
    # 3. Count similar pairs using a threshold

    print("Encoding queries...")
    with torch.no_grad():
        # Encode attack queries
        attack_embeddings = []
        batch_size = 50
        for i in range(0, len(attack_subset), batch_size):
            batch = attack_subset[i : min(i + batch_size, len(attack_subset))].to(
                device
            )
            # Get item embeddings
            item_embeddings = model.item_embeddings(batch)
            # Encode using the similarity encoder
            emb = encoder(item_embeddings).cpu()
            attack_embeddings.append(emb)
        attack_embeddings = torch.cat(attack_embeddings, dim=0)

        # Encode benign queries
        benign_embeddings = []
        for i in range(0, len(benign_queries), batch_size):
            batch = benign_queries[i : min(i + batch_size, len(benign_queries))].to(
                device
            )
            # Get item embeddings
            item_embeddings = model.item_embeddings(batch)
            # Encode using the similarity encoder
            emb = encoder(item_embeddings).cpu()
            benign_embeddings.append(emb)
        benign_embeddings = torch.cat(benign_embeddings, dim=0)

    print("Computing pairwise distances for attack queries...")
    attack_distances = []
    # Consider up to 10,000 pairs to prevent memory issues
    max_pairs = 10000
    count = 0

    # Simulate the account-based detection scenario
    # Assume queries within the same account come in sequence
    sequence_length = 100  # queries per account
    num_sequences = len(attack_subset) // sequence_length

    attack_similar_pairs_counts = []

    # For each simulated account
    for seq_idx in range(num_sequences):
        start_idx = seq_idx * sequence_length
        end_idx = min((seq_idx + 1) * sequence_length, len(attack_embeddings))
        seq_embeddings = attack_embeddings[start_idx:end_idx]

        similar_pairs = 0
        # Count pairs within this sequence
        for i in range(len(seq_embeddings)):
            for j in range(i + 1, len(seq_embeddings)):
                # Use L2 distance as similarity measure
                dist = torch.norm(seq_embeddings[i] - seq_embeddings[j], p=2).item()
                attack_distances.append(dist)
                count += 1
                if count >= max_pairs:
                    break
            if count >= max_pairs:
                break

        # Store number of similar pairs per account (will compute with threshold later)
        attack_similar_pairs_counts.append(
            len(seq_embeddings) * (len(seq_embeddings) - 1) // 2
        )

    print("Computing pairwise distances for benign queries...")
    benign_distances = []
    count = 0

    benign_similar_pairs_counts = []

    # For each simulated benign account
    for seq_idx in range(num_sequences):
        start_idx = seq_idx * sequence_length
        end_idx = min((seq_idx + 1) * sequence_length, len(benign_embeddings))
        seq_embeddings = benign_embeddings[start_idx:end_idx]

        similar_pairs = 0
        # Count pairs within this sequence
        for i in range(len(seq_embeddings)):
            for j in range(i + 1, len(seq_embeddings)):
                # Use L2 distance as similarity measure
                dist = torch.norm(seq_embeddings[i] - seq_embeddings[j], p=2).item()
                benign_distances.append(dist)
                count += 1
                if count >= max_pairs:
                    break
            if count >= max_pairs:
                break

        # Store number of similar pairs per account (will compute with threshold later)
        benign_similar_pairs_counts.append(
            len(seq_embeddings) * (len(seq_embeddings) - 1) // 2
        )

    # Plot distance distributions
    plt.figure(figsize=(10, 6))
    plt.hist(attack_distances, bins=50, alpha=0.5, label="Attack Pairs", density=True)
    plt.hist(benign_distances, bins=50, alpha=0.5, label="Benign Pairs", density=True)
    plt.xlabel("L2 Distance")
    plt.ylabel("Density")
    plt.title("Distribution of Pairwise Distances")
    plt.legend()
    plt.savefig("figures/seat/distance_distributions.png", dpi=300, bbox_inches="tight")

    # Determine optimal similarity threshold
    # Compute ROC curve for various distance thresholds
    min_dist = min(min(attack_distances), min(benign_distances))
    max_dist = max(max(attack_distances), max(benign_distances))
    thresholds = np.linspace(min_dist, max_dist, 100)

    # True labels for the distances (1 for attack, 0 for benign)
    y_true = np.concatenate(
        [np.ones(len(attack_distances)), np.zeros(len(benign_distances))]
    )
    distances = np.concatenate([attack_distances, benign_distances])

    tprs = []
    fprs = []

    # Calculate TPR and FPR for each threshold
    for threshold in thresholds:
        # Predict as attack if distance is below threshold (similar pairs)
        y_pred = (distances <= threshold).astype(int)

        # True positive rate
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        # False positive rate
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    # Find threshold that gives target FPR
    target_fpr_idx = np.argmin(np.abs(np.array(fprs) - args.target_fpr))
    optimal_similarity_threshold = thresholds[target_fpr_idx]

    # For the balanced approach, we want to choose a threshold that maximizes TPR-FPR
    balanced_idx = np.argmax(np.array(tprs) - np.array(fprs))
    balanced_threshold = thresholds[balanced_idx]

    print(
        f"Optimal similarity threshold for target FPR: {optimal_similarity_threshold:.4f}"
    )
    print(f"Balanced threshold (max TPR-FPR): {balanced_threshold:.4f}")

    # Now compute actual similar pairs for each account based on the selected threshold
    attack_similar_pairs = []
    for seq_idx in range(num_sequences):
        start_idx = seq_idx * sequence_length
        end_idx = min((seq_idx + 1) * sequence_length, len(attack_embeddings))
        seq_embeddings = attack_embeddings[start_idx:end_idx]

        similar_pairs = 0
        for i in range(len(seq_embeddings)):
            for j in range(i + 1, len(seq_embeddings)):
                dist = torch.norm(seq_embeddings[i] - seq_embeddings[j], p=2).item()
                if dist <= balanced_threshold:
                    similar_pairs += 1

        attack_similar_pairs.append(similar_pairs)

    benign_similar_pairs = []
    for seq_idx in range(num_sequences):
        start_idx = seq_idx * sequence_length
        end_idx = min((seq_idx + 1) * sequence_length, len(benign_embeddings))
        seq_embeddings = benign_embeddings[start_idx:end_idx]

        similar_pairs = 0
        for i in range(len(seq_embeddings)):
            for j in range(i + 1, len(seq_embeddings)):
                dist = torch.norm(seq_embeddings[i] - seq_embeddings[j], p=2).item()
                if dist <= balanced_threshold:
                    similar_pairs += 1

        benign_similar_pairs.append(similar_pairs)

    # Determine thresholds based on the data distributions
    # Use percentile-based thresholding for better separation
    benign_percentile = (
        np.percentile(benign_similar_pairs, 95) if benign_similar_pairs else 0
    )
    attack_percentile = (
        np.percentile(attack_similar_pairs, 5) if attack_similar_pairs else 0
    )

    # Use the midpoint as a balanced threshold
    balanced_pairs_threshold = int((benign_percentile + attack_percentile) / 2)
    # Ensure we have a reasonable minimum threshold
    balanced_pairs_threshold = max(5, balanced_pairs_threshold)

    # For the aggressive approach, use a lower threshold (75th percentile of benign)
    aggressive_threshold = (
        int(np.percentile(benign_similar_pairs, 75)) if benign_similar_pairs else 0
    )
    aggressive_threshold = max(3, aggressive_threshold)

    print(f"Data-driven thresholds:")
    print(f"95th percentile of benign similar pairs: {benign_percentile:.1f}")
    print(f"5th percentile of attack similar pairs: {attack_percentile:.1f}")
    print(f"Balanced threshold: {balanced_pairs_threshold}")
    print(f"Aggressive threshold: {aggressive_threshold}")

    # Evaluate detection performance for both approaches
    attack_detection = [
        pairs > balanced_pairs_threshold for pairs in attack_similar_pairs
    ]
    benign_detection = [
        pairs > balanced_pairs_threshold for pairs in benign_similar_pairs
    ]

    attack_detection_aggressive = [
        pairs > aggressive_threshold for pairs in attack_similar_pairs
    ]
    benign_detection_aggressive = [
        pairs > aggressive_threshold for pairs in benign_similar_pairs
    ]

    attack_detection_rate = (
        sum(attack_detection) / len(attack_detection) if attack_detection else 0
    )
    false_positive_rate = (
        sum(benign_detection) / len(benign_detection) if benign_detection else 0
    )

    attack_detection_rate_aggressive = (
        sum(attack_detection_aggressive) / len(attack_detection_aggressive)
        if attack_detection_aggressive
        else 0
    )
    false_positive_rate_aggressive = (
        sum(benign_detection_aggressive) / len(benign_detection_aggressive)
        if benign_detection_aggressive
        else 0
    )

    # Calculate accounts needed for both approaches
    accounts_needed = (
        int(1 / (1 - attack_detection_rate) + 1) if attack_detection_rate < 1 else 10
    )
    accounts_needed_aggressive = (
        int(1 / (1 - attack_detection_rate_aggressive) + 1)
        if attack_detection_rate_aggressive < 1
        else 10
    )

    # Plot similar pairs distributions
    plt.figure(figsize=(10, 6))
    plt.hist(
        attack_similar_pairs, bins=20, alpha=0.5, label="Attack Accounts", density=True
    )
    plt.hist(
        benign_similar_pairs, bins=20, alpha=0.5, label="Benign Accounts", density=True
    )
    plt.axvline(
        x=balanced_pairs_threshold,
        color="r",
        linestyle="--",
        label=f"Threshold = {balanced_pairs_threshold}",
    )
    plt.axvline(
        x=aggressive_threshold,
        color="g",
        linestyle="--",
        label=f"Aggressive = {aggressive_threshold}",
    )
    plt.xlabel("Number of Similar Pairs")
    plt.ylabel("Density")
    plt.title("Distribution of Similar Pairs per Account")
    plt.legend()
    plt.savefig(
        "figures/seat/similar_pairs_distributions.png", dpi=300, bbox_inches="tight"
    )

    # Plot confusion matrix
    cm = confusion_matrix(
        [1] * len(attack_detection) + [0] * len(benign_detection),
        attack_detection + benign_detection,
    )

    plt.figure(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Benign", "Attack"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("SEAT Defense: Confusion Matrix", fontsize=14)
    plt.savefig("figures/seat/confusion_matrix.png", dpi=300, bbox_inches="tight")

    # Update the print results section to show both approaches
    print("\nSEAT Defense Results:")
    print("------------------------------")
    print(f"Standard Approach (Pairs Threshold: {balanced_pairs_threshold})")
    print(f"Attack Detection Rate: {attack_detection_rate:.2%}")
    print(f"False Positive Rate: {false_positive_rate:.2%}")
    print(f"Accounts Needed for Attack: {accounts_needed}")
    print(f"Similarity Threshold: {balanced_threshold:.4f}")
    print(f"\nAggressive Approach (Pairs Threshold: {aggressive_threshold})")
    print(f"Attack Detection Rate: {attack_detection_rate_aggressive:.2%}")
    print(f"False Positive Rate: {false_positive_rate_aggressive:.2%}")
    print(f"Accounts Needed for Attack: {accounts_needed_aggressive}")
    print(f"Similarity Threshold: {balanced_threshold:.4f}")

    # Update the results dict to include both approaches
    results = {
        "standard": {
            "attack_detection_rate": float(attack_detection_rate),
            "false_positive_rate": float(false_positive_rate),
            "similarity_threshold": float(balanced_threshold),
            "pairs_threshold": int(balanced_pairs_threshold),
            "accounts_needed": int(accounts_needed),
        },
        "aggressive": {
            "attack_detection_rate": float(attack_detection_rate_aggressive),
            "false_positive_rate": float(false_positive_rate_aggressive),
            "similarity_threshold": float(balanced_threshold),
            "pairs_threshold": int(aggressive_threshold),
            "accounts_needed": int(accounts_needed_aggressive),
        },
    }

    torch.save(results, os.path.join(args.output_dir, "seat_results.pt"))
    print(f"\nResults saved to {args.output_dir}")
    print(f"Figures saved to figures/seat/")


if __name__ == "__main__":
    main()
