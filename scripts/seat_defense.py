import argparse
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
        # Apply encoder without normalization (paper uses L2 distance directly)
        return self.encoder(x)


def main():
    parser = argparse.ArgumentParser(
        description="SEAT defense implementation as described in the original paper"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--encoder_path", type=str, required=True)
    parser.add_argument("--attack_data", type=str, required=True)
    parser.add_argument("--benign_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="defense_results")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
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

    # Use all attack data
    attack_subset = attack_queries
    print(
        f"Using {len(attack_subset)} attack queries and {len(benign_queries)} benign queries"
    )

    # STEP 1: Encode all queries
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

    # STEP 2: Find similarity threshold with low FPR
    # Compute all pairwise distances between benign queries to calibrate threshold
    print("Computing benign similarity distribution...")
    benign_distances = []
    max_sample = min(1000, len(benign_embeddings))  # Limit sample size for efficiency
    for i in range(max_sample):
        for j in range(i + 1, max_sample):
            dist = torch.norm(benign_embeddings[i] - benign_embeddings[j], p=2).item()
            benign_distances.append(dist)

    # Sort distances and find threshold for desired FPR (0.01% as in paper)
    benign_distances.sort()
    total_pairs = len(benign_distances)
    fpr_idx = max(1, int(0.0001 * total_pairs))
    similarity_threshold = benign_distances[fpr_idx]

    print(f"Similarity threshold for 0.01% FPR: {similarity_threshold:.4f}")

    # Add more granular threshold analysis
    similarity_test_thresholds = [
        similarity_threshold * 0.5,
        similarity_threshold * 0.8,
        similarity_threshold,
        similarity_threshold * 1.2,
        similarity_threshold * 1.5,
    ]
    print("\nTesting different similarity thresholds:")
    for sim_threshold in similarity_test_thresholds:
        benign_fp_rate = sum(1 for d in benign_distances if d < sim_threshold) / len(
            benign_distances
        )
        print(
            f"Similarity threshold {sim_threshold:.4f} → benign FP rate: {benign_fp_rate:.4%}"
        )

    # STEP 4: Count similar pairs per account with more realistic scenarios
    print("\nAnalyzing similar pairs with different account configurations...")

    # Scenario 1: All attack queries in one account (original analysis)
    print("\nScenario 1: All attack queries from single account")
    attack_similar_pairs_s1 = []
    similar_pairs = 0
    for i in range(len(attack_embeddings)):
        for j in range(i + 1, len(attack_embeddings)):
            dist = torch.norm(attack_embeddings[i] - attack_embeddings[j], p=2).item()
            if dist < similarity_threshold:
                similar_pairs += 1

    attack_similar_pairs_s1.append(similar_pairs)
    print(f"Attack account (all queries): {similar_pairs} similar pairs")

    # Scenario 2: Attack queries split into multiple smaller accounts
    print("\nScenario 2: Attack split across multiple accounts (100 queries each)")
    attack_similar_pairs_s2 = []
    attack_sequence_length = 100
    num_attack_sequences = len(attack_embeddings) // attack_sequence_length

    for seq_idx in range(num_attack_sequences):
        start_idx = seq_idx * attack_sequence_length
        end_idx = min((seq_idx + 1) * attack_sequence_length, len(attack_embeddings))
        seq_embeddings = attack_embeddings[start_idx:end_idx]

        similar_pairs = 0
        for i in range(len(seq_embeddings)):
            for j in range(i + 1, len(seq_embeddings)):
                dist = torch.norm(seq_embeddings[i] - seq_embeddings[j], p=2).item()
                if dist < similarity_threshold:
                    similar_pairs += 1

        attack_similar_pairs_s2.append(similar_pairs)

    print(f"Attack accounts (100 queries each): {attack_similar_pairs_s2}")
    print(f"Max similar pairs: {max(attack_similar_pairs_s2)}")
    print(f"Min similar pairs: {min(attack_similar_pairs_s2)}")
    print(
        f"Average similar pairs: {sum(attack_similar_pairs_s2)/len(attack_similar_pairs_s2):.2f}"
    )

    # Scenario 3: Simulate query partitioning attack (distribute pairs across accounts)
    print("\nScenario 3: Simulated query partitioning attack")
    # In this strategy, attacker avoids putting similar queries in same account
    attack_similar_pairs_s3 = []
    # Simple simulation: distribute seed image queries across accounts
    num_accounts = 10
    account_queries = [[] for _ in range(num_accounts)]

    # Simple query partitioning simulation
    for i, emb in enumerate(attack_embeddings):
        account_idx = i % num_accounts
        account_queries[account_idx].append(emb)

    for account_idx in range(num_accounts):
        account_embs = account_queries[account_idx]
        if len(account_embs) < 2:
            continue

        similar_pairs = 0
        account_embs = torch.stack(account_embs)
        for i in range(len(account_embs)):
            for j in range(i + 1, len(account_embs)):
                dist = torch.norm(account_embs[i] - account_embs[j], p=2).item()
                if dist < similarity_threshold:
                    similar_pairs += 1

        attack_similar_pairs_s3.append(similar_pairs)

    if attack_similar_pairs_s3:
        print(f"Attack accounts (partitioned): {attack_similar_pairs_s3}")
        print(f"Max similar pairs: {max(attack_similar_pairs_s3)}")
        print(f"Min similar pairs: {min(attack_similar_pairs_s3)}")
        print(
            f"Average similar pairs: {sum(attack_similar_pairs_s3)/len(attack_similar_pairs_s3):.2f}"
        )
    else:
        print("No attack accounts with >1 query in partitioning simulation")

    # For benign, use multiple smaller accounts
    sequence_length = 100  # queries per benign account
    num_benign_sequences = len(benign_embeddings) // sequence_length

    # Count similar pairs for benign accounts
    benign_similar_pairs = []
    for seq_idx in range(num_benign_sequences):
        start_idx = seq_idx * sequence_length
        end_idx = min((seq_idx + 1) * sequence_length, len(benign_embeddings))
        seq_embeddings = benign_embeddings[start_idx:end_idx]

        similar_pairs = 0
        for i in range(len(seq_embeddings)):
            for j in range(i + 1, len(seq_embeddings)):
                dist = torch.norm(seq_embeddings[i] - seq_embeddings[j], p=2).item()
                if dist < similarity_threshold:
                    similar_pairs += 1

        benign_similar_pairs.append(similar_pairs)

    print(f"\nBenign accounts:")
    print(
        f"Max benign similar pairs: {max(benign_similar_pairs) if benign_similar_pairs else 0}"
    )
    print(
        f"Avg benign similar pairs: {sum(benign_similar_pairs)/len(benign_similar_pairs) if benign_similar_pairs else 0:.2f}"
    )

    # Try different similar pair thresholds for detection
    pairs_thresholds = [5, 10, 20, 30, 40, 50, 75, 100]
    print("\nEvaluating detection performance with different thresholds and scenarios:")
    print("Pairs Threshold | Scenario 1  | Scenario 2  | Scenario 3  | False Positive")
    print("---------------|-------------|-------------|-------------|-------------")

    for threshold in pairs_thresholds:
        # Calculate detection rates for each scenario
        attack_detection_s1 = [pairs > threshold for pairs in attack_similar_pairs_s1]
        attack_detection_s2 = [pairs > threshold for pairs in attack_similar_pairs_s2]
        attack_detection_s3 = (
            [pairs > threshold for pairs in attack_similar_pairs_s3]
            if attack_similar_pairs_s3
            else []
        )
        benign_detection = [pairs > threshold for pairs in benign_similar_pairs]

        attack_detection_rate_s1 = (
            sum(attack_detection_s1) / len(attack_detection_s1)
            if attack_detection_s1
            else 0
        )
        attack_detection_rate_s2 = (
            sum(attack_detection_s2) / len(attack_detection_s2)
            if attack_detection_s2
            else 0
        )
        attack_detection_rate_s3 = (
            sum(attack_detection_s3) / len(attack_detection_s3)
            if attack_detection_s3
            else 0
        )
        false_positive_rate = (
            sum(benign_detection) / len(benign_detection) if benign_detection else 0
        )

        print(
            f"{threshold:14d} | {attack_detection_rate_s1:11.2%} | {attack_detection_rate_s2:11.2%} | {attack_detection_rate_s3:11.2%} | {false_positive_rate:11.2%}"
        )

    # Use the SEAT paper's threshold (50)
    pairs_threshold = 50

    # STEP 5: Detection using the original scenario (all attack queries from 1 account)
    attack_detection = [pairs > pairs_threshold for pairs in attack_similar_pairs_s1]
    benign_detection = [pairs > pairs_threshold for pairs in benign_similar_pairs]

    attack_detection_rate = (
        sum(attack_detection) / len(attack_detection) if attack_detection else 0
    )
    false_positive_rate = (
        sum(benign_detection) / len(benign_detection) if benign_detection else 0
    )

    # Calculate accounts needed for attack as in the paper
    accounts_needed = (
        int(1 / (1 - attack_detection_rate) + 1) if attack_detection_rate < 1 else 10
    )

    # Plot similar pairs distributions
    plt.figure(figsize=(10, 6))
    plt.hist(
        attack_similar_pairs_s1,
        bins=20,
        alpha=0.5,
        label="Attack Accounts",
        density=True,
    )
    plt.hist(
        benign_similar_pairs, bins=20, alpha=0.5, label="Benign Accounts", density=True
    )
    plt.axvline(
        x=pairs_threshold,
        color="r",
        linestyle="--",
        label=f"Threshold = {pairs_threshold}",
    )
    plt.xlabel("Number of Similar Pairs")
    plt.ylabel("Density")
    plt.title("Distribution of Similar Pairs per Account")
    plt.legend()
    plt.savefig(
        "figures/seat/similar_pairs_distributions.png", dpi=300, bbox_inches="tight"
    )

    # Visualize L2 distances
    plt.figure(figsize=(10, 6))

    # Sample a subset of attack and benign distances for visualization
    attack_sample_distances = []
    for seq_embeddings in [
        attack_embeddings[i : i + sequence_length]
        for i in range(0, len(attack_embeddings), sequence_length)
    ][:5]:
        for i in range(min(20, len(seq_embeddings))):
            for j in range(i + 1, min(20, len(seq_embeddings))):
                dist = torch.norm(seq_embeddings[i] - seq_embeddings[j], p=2).item()
                attack_sample_distances.append(dist)

    benign_sample_distances = []
    for seq_embeddings in [
        benign_embeddings[i : i + sequence_length]
        for i in range(0, len(benign_embeddings), sequence_length)
    ][:5]:
        for i in range(min(20, len(seq_embeddings))):
            for j in range(i + 1, min(20, len(seq_embeddings))):
                dist = torch.norm(seq_embeddings[i] - seq_embeddings[j], p=2).item()
                benign_sample_distances.append(dist)

    plt.hist(attack_sample_distances, bins=50, alpha=0.5, label="Attack", density=True)
    plt.hist(benign_sample_distances, bins=50, alpha=0.5, label="Benign", density=True)
    plt.axvline(
        x=similarity_threshold,
        color="r",
        linestyle="--",
        label=f"Similarity Threshold = {similarity_threshold:.4f}",
    )
    plt.xlabel("L2 Distance")
    plt.ylabel("Density")
    plt.title("Distribution of L2 Distances Between Query Embeddings")
    plt.legend()
    plt.savefig("figures/seat/l2_distributions.png", dpi=300, bbox_inches="tight")

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

    # Print results
    print("\nSEAT Defense Results:")
    print("------------------------------")
    print(f"Attack Detection Rate: {attack_detection_rate:.2%}")
    print(f"False Positive Rate: {false_positive_rate:.2%}")
    print(f"Accounts Needed for Attack: {accounts_needed}")
    print(f"Similarity Threshold: {similarity_threshold:.4f}")
    print(f"Similar Pairs Threshold: {pairs_threshold}")

    # Save results
    results = {
        "attack_detection_rate": float(attack_detection_rate),
        "false_positive_rate": float(false_positive_rate),
        "similarity_threshold": float(similarity_threshold),
        "pairs_threshold": int(pairs_threshold),
        "accounts_needed": int(accounts_needed),
    }

    torch.save(results, os.path.join(args.output_dir, "seat_results.pt"))
    print(f"Results saved to {args.output_dir}")
    print(f"Figures saved to figures/seat/")


if __name__ == "__main__":
    main()
