#!/usr/bin/env python
"""
Run SEAT defense to detect model extraction attacks.

This script loads a trained similarity encoder and runs the SEAT defense
on a dataset of attack queries to measure its effectiveness.
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.defense.SEAT import (
    SimilarityEncoder,
    SEATDetector,
    compute_accounts_needed,
    visualize_distances,
    visualize_similar_pairs,
    visualize_confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run SEAT defense against model extraction attacks"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to defend"
    )
    parser.add_argument(
        "--encoder_path", type=str, required=True, help="Path to trained SEAT encoder"
    )
    parser.add_argument(
        "--attack_data", type=str, required=True, help="Path to attack queries data"
    )
    parser.add_argument(
        "--benign_data", type=str, required=True, help="Path to benign queries data"
    )
    parser.add_argument(
        "--pairs_threshold",
        type=int,
        default=50,
        help="Threshold for number of similar pairs to flag an account",
    )
    parser.add_argument(
        "--fpr_target",
        type=float,
        default=0.0001,
        help="Target false positive rate (default 0.01%)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="defense_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Batch size for processing"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--account_size",
        type=int,
        default=100,
        help="Number of queries per simulated account",
    )
    parser.add_argument(
        "--query_partitioning",
        action="store_true",
        help="Simulate query partitioning attack (distribute similar queries across accounts)",
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    from show_recommendations import load_model

    model = load_model(args.model_path, 3953, 256, device)

    # Load trained encoder
    encoder_data = torch.load(args.encoder_path)
    encoder = SimilarityEncoder(256, 128)
    encoder.load_state_dict(
        encoder_data["state_dict"] if "state_dict" in encoder_data else encoder_data
    )
    encoder.to(device)
    encoder.eval()

    # Load query data
    attack_queries = torch.load(args.attack_data)
    benign_dataset = torch.load(args.benign_data)

    # For benign data, take the first element if it's a tuple
    # (data might be in the form (item_id, label))
    benign_queries_list = []
    for i in range(min(len(benign_dataset), 1000)):
        item = benign_dataset[i]
        if isinstance(item, tuple):
            benign_queries_list.append(item[0])
        else:
            benign_queries_list.append(item)

    benign_queries = torch.stack(benign_queries_list)

    print(
        f"Loaded {len(attack_queries)} attack queries and {len(benign_queries)} benign queries"
    )

    # Create SEAT detector
    seat_detector = SEATDetector(
        encoder=encoder,
        pairs_threshold=args.pairs_threshold,
        fpr_target=args.fpr_target,
    )

    # Step 1: Encode all queries
    print("Encoding queries...")
    with torch.no_grad():
        # Encode attack queries
        attack_embeddings = []
        batch_size = args.batch_size
        for i in range(0, len(attack_queries), batch_size):
            # Handle batch extraction and device transfer
            batch_indices = range(i, min(i + batch_size, len(attack_queries)))

            if isinstance(attack_queries, torch.Tensor):
                batch = attack_queries[batch_indices].to(device)
            else:
                batch_list = [attack_queries[j] for j in batch_indices]
                if all(isinstance(item, torch.Tensor) for item in batch_list):
                    batch = torch.stack(batch_list).to(device)
                else:
                    print(f"Warning: Skipping batch with non-tensor items")
                    continue

            # Get item embeddings
            item_embeddings = model.item_embeddings(batch)
            # Encode using similarity encoder
            emb = encoder(item_embeddings).cpu()
            attack_embeddings.append(emb)

        if attack_embeddings:
            attack_embeddings = torch.cat(attack_embeddings, dim=0)
        else:
            print("Error: No valid attack embeddings could be created")
            return

        # Encode benign queries
        benign_embeddings = []
        for i in range(0, len(benign_queries), batch_size):
            # Handle batch extraction and device transfer
            batch_indices = range(i, min(i + batch_size, len(benign_queries)))

            if isinstance(benign_queries, torch.Tensor):
                batch = benign_queries[batch_indices].to(device)
            else:
                batch_list = [benign_queries[j] for j in batch_indices]
                if all(isinstance(item, torch.Tensor) for item in batch_list):
                    batch = torch.stack(batch_list).to(device)
                else:
                    print(f"Warning: Skipping batch with non-tensor items")
                    continue

            # Get item embeddings
            item_embeddings = model.item_embeddings(batch)
            # Encode using similarity encoder
            emb = encoder(item_embeddings).cpu()
            benign_embeddings.append(emb)

        if benign_embeddings:
            benign_embeddings = torch.cat(benign_embeddings, dim=0)
        else:
            print("Error: No valid benign embeddings could be created")
            return

    # Calibrate similarity threshold based on distribution of benign similarities
    similarities = []
    for i in range(len(benign_embeddings)):
        for j in range(i + 1, len(benign_embeddings)):
            dist = torch.sum((benign_embeddings[i] - benign_embeddings[j]) ** 2).item()
            similarities.append(dist)

    # Use different account size for more granular testing
    # While keeping original thresholds
    args.account_size = 20  # Keep this for more granular testing

    # Restore original similarity threshold calculation
    # Use the 5th percentile for similarity threshold (original value)
    similarity_threshold = np.percentile(similarities, 5)
    print(f"Calibrated similarity threshold: {similarity_threshold:.4f}")

    # Set the calculated threshold in the detector
    seat_detector.similarity_threshold = similarity_threshold

    # Restore the original pairs threshold
    # Using the paper's standard value of 50
    args.pairs_threshold = 50

    # Use more discriminative thresholds for detection
    pairs_thresholds = [3, 5, 10, 15, 20, 30, 50, 100]

    # Step 3: Simulate different account scenarios
    print("\nSimulating different account scenarios...")

    # Scenario 1: All attack queries in one account
    print("\nScenario 1: All attack queries from single account")
    seat_detector.reset()
    attack_similar_pairs_s1 = []
    account_id = "attack_account_all"

    # Find similar query pairs for Scenario 1
    num_similar_pairs = 0
    similar_pairs = []

    for i in tqdm(range(len(attack_embeddings))):
        for j in range(i + 1, len(attack_embeddings)):
            # Use squared Euclidean distance for efficiency
            dist = torch.sum((attack_embeddings[i] - attack_embeddings[j]) ** 2).item()

            # If distance is below threshold, queries are suspiciously similar
            if dist < similarity_threshold:
                similar_pairs.append((i, j, dist))
                num_similar_pairs += 1

                # Also record this in the detector for consistency
                if account_id not in seat_detector.account_similar_pairs:
                    seat_detector.account_similar_pairs[account_id] = 0
                seat_detector.account_similar_pairs[account_id] += 1

    print(f"Attack account (all queries): {num_similar_pairs} similar pairs")

    # Additional metric: Calculate density of similar pairs (clusters)
    if len(similar_pairs) > 0:
        # Create graph of similar pairs
        G = nx.Graph()
        for i, j, _ in similar_pairs:
            G.add_edge(i, j)

        # Find connected components (clusters)
        clusters = list(nx.connected_components(G))
        largest_cluster = max(len(c) for c in clusters) if clusters else 0

        print(f"Number of suspicious clusters: {len(clusters)}")
        print(f"Largest cluster size: {largest_cluster} queries")

        # Higher cluster density indicates more organized attack
        if largest_cluster > 10:
            print("ALERT: High-density attack pattern detected!")

    attack_similar_pairs_s1.append(seat_detector.account_similar_pairs[account_id])
    print(f"Attack account (all queries): {attack_similar_pairs_s1[0]} similar pairs")

    # Scenario 2: Attack queries split into smaller accounts
    print("\nScenario 2: Attack split across multiple accounts")
    seat_detector.reset()
    attack_similar_pairs_s2 = []
    account_size = args.account_size
    num_accounts = len(attack_embeddings) // account_size + (
        1 if len(attack_embeddings) % account_size > 0 else 0
    )

    account_detection_status = [False] * num_accounts
    for i, emb in enumerate(tqdm(attack_embeddings)):
        account_idx = i // account_size
        account_id = f"attack_account_{account_idx}"

        if account_detection_status[account_idx]:
            # Skip processing for already flagged accounts
            continue

        result = seat_detector.process_query(emb, account_id)
        if result:
            account_detection_status[account_idx] = True

    # Get similar pairs for each account
    for account_idx in range(num_accounts):
        account_id = f"attack_account_{account_idx}"
        if account_id in seat_detector.account_similar_pairs:
            attack_similar_pairs_s2.append(
                seat_detector.account_similar_pairs[account_id]
            )
        else:
            attack_similar_pairs_s2.append(0)

    accounts_flagged = sum(account_detection_status)
    print(
        f"Flagged {accounts_flagged} accounts out of {num_accounts} ({accounts_flagged/num_accounts:.2%})"
    )
    print(f"Max similar pairs: {max(attack_similar_pairs_s2)}")
    print(
        f"Avg similar pairs: {sum(attack_similar_pairs_s2)/len(attack_similar_pairs_s2):.2f}"
    )

    # Scenario 3: Query partitioning attack (if enabled)
    attack_similar_pairs_s3 = []
    if args.query_partitioning:
        print("\nScenario 3: Simulated query partitioning attack")
        seat_detector.reset()

        # Simulate query partitioning: distribute queries across accounts to avoid similar pairs
        num_partition_accounts = 10
        account_queries = [[] for _ in range(num_partition_accounts)]

        # Simple simulation: distribute queries using modulo
        for i, emb in enumerate(attack_embeddings):
            account_idx = i % num_partition_accounts
            account_queries[account_idx].append(emb)

        # Process queries for each account
        account_detection_status = [False] * num_partition_accounts
        for account_idx in range(num_partition_accounts):
            account_id = f"attack_partition_{account_idx}"
            account_embs = account_queries[account_idx]

            for emb in account_embs:
                if account_detection_status[account_idx]:
                    # Skip processing for already flagged accounts
                    continue

                result = seat_detector.process_query(emb, account_id)
                if result:
                    account_detection_status[account_idx] = True

            if account_id in seat_detector.account_similar_pairs:
                attack_similar_pairs_s3.append(
                    seat_detector.account_similar_pairs[account_id]
                )
            else:
                attack_similar_pairs_s3.append(0)

        accounts_flagged = sum(account_detection_status)
        print(
            f"Flagged {accounts_flagged} accounts out of {num_partition_accounts} ({accounts_flagged/num_partition_accounts:.2%})"
        )
        print(f"Max similar pairs: {max(attack_similar_pairs_s3)}")
        print(
            f"Avg similar pairs: {sum(attack_similar_pairs_s3)/len(attack_similar_pairs_s3):.2f}"
        )

    # For benign, use multiple accounts
    print("\nProcessing benign queries...")
    seat_detector.reset()
    benign_similar_pairs = []
    benign_detection_status = []

    # Use more benign accounts to better estimate FPR
    # We'll create 50 accounts with 20 queries each
    num_benign_accounts = 50

    # Process benign queries in account-sized chunks
    for account_idx in range(
        min(num_benign_accounts, len(benign_embeddings) // args.account_size)
    ):
        account_id = f"benign_account_{account_idx}"
        account_detected = False

        start_idx = account_idx * args.account_size
        end_idx = min((account_idx + 1) * args.account_size, len(benign_embeddings))

        for i in range(start_idx, end_idx):
            if account_detected:
                break

            result = seat_detector.process_query(benign_embeddings[i], account_id)
            if result:
                account_detected = True
                print(f"Benign account {account_id} flagged as suspicious!")

        benign_detection_status.append(account_detected)

        if account_id in seat_detector.account_similar_pairs:
            benign_similar_pairs.append(seat_detector.account_similar_pairs[account_id])
        else:
            benign_similar_pairs.append(0)

    # Count false positives
    flagged_benign = sum(benign_detection_status)
    print(
        f"Flagged {flagged_benign} out of {len(benign_detection_status)} benign accounts"
    )

    false_positive_rate = (
        flagged_benign / len(benign_detection_status) if benign_detection_status else 0
    )
    print(f"False positive rate: {false_positive_rate:.6%}")
    print(
        f"Max benign similar pairs: {max(benign_similar_pairs) if benign_similar_pairs else 0}"
    )
    print(
        f"Avg benign similar pairs: {sum(benign_similar_pairs)/len(benign_similar_pairs) if benign_similar_pairs else 0:.2f}"
    )

    # Print details about similar pairs distribution in benign accounts
    print("\nDistribution of similar pairs in benign accounts:")
    pair_counts = {i: 0 for i in range(0, max(benign_similar_pairs) + 1, 5)}
    for count in benign_similar_pairs:
        # Round down to nearest 5
        bucket = (count // 5) * 5
        if bucket in pair_counts:
            pair_counts[bucket] += 1

    for count, num_accounts in sorted(pair_counts.items()):
        if num_accounts > 0:
            print(f"  {count}-{count+4} pairs: {num_accounts} accounts")

    # Try different thresholds
    print("\n" + "=" * 60)
    print("                  SEAT DEFENSE RESULTS")
    print("=" * 60)

    # Set the pairs threshold
    pairs_threshold = args.pairs_threshold

    # Detection for scenario 1 (all queries in one account)
    attack_detection = [pairs > pairs_threshold for pairs in attack_similar_pairs_s1]
    benign_detection = [pairs > pairs_threshold for pairs in benign_similar_pairs]

    attack_detection_rate = (
        sum(attack_detection) / len(attack_detection) if attack_detection else 0
    )
    false_positive_rate = (
        sum(benign_detection) / len(benign_detection) if benign_detection else 0
    )

    # Calculate accounts needed for attack as in the paper
    accounts_needed = compute_accounts_needed(attack_detection_rate)

    # Visualizations
    # 1. Distances distribution
    attack_sample_distances = []
    benign_sample_distances = []

    # Sample attack distances
    for i in range(min(100, len(attack_embeddings))):
        for j in range(i + 1, min(100, len(attack_embeddings))):
            dist = torch.norm(attack_embeddings[i] - attack_embeddings[j], p=2).item()
            attack_sample_distances.append(dist)

    # Sample benign distances
    for i in range(min(100, len(benign_embeddings))):
        for j in range(i + 1, min(100, len(benign_embeddings))):
            dist = torch.norm(benign_embeddings[i] - benign_embeddings[j], p=2).item()
            benign_sample_distances.append(dist)

    visualize_distances(
        attack_distances=attack_sample_distances,
        benign_distances=benign_sample_distances,
        similarity_threshold=similarity_threshold,
        save_path=os.path.join(args.output_dir, "figures", "l2_distributions.png"),
    )

    # 2. Similar pairs distribution
    visualize_similar_pairs(
        attack_pairs=attack_similar_pairs_s1 if attack_similar_pairs_s1 else [0],
        benign_pairs=benign_similar_pairs,
        pairs_threshold=pairs_threshold,
        save_path=os.path.join(
            args.output_dir, "figures", "similar_pairs_distributions.png"
        ),
    )

    # 3. Confusion matrix
    visualize_confusion_matrix(
        attack_detection=attack_detection,
        benign_detection=benign_detection,
        save_path=os.path.join(args.output_dir, "figures", "confusion_matrix.png"),
    )

    # Print results in a more visually appealing format
    print("\n" + "-" * 60)
    print("  ATTACK DETECTION STATISTICS")
    print("-" * 60)
    print(f"▶ Attack Detection Rate:     {attack_detection_rate:.2%}")
    print(f"▶ False Positive Rate:       {false_positive_rate:.4%}")
    print(f"▶ Accounts Needed for Attack: {accounts_needed}")

    print("\n" + "-" * 60)
    print("  SYSTEM CONFIGURATION")
    print("-" * 60)
    print(f"▶ Similarity Threshold:     {similarity_threshold:.4f}")
    print(f"▶ Similar Pairs Threshold:  {pairs_threshold}")

    print("\n" + "-" * 60)
    print("  ATTACK ACCOUNT PATTERNS")
    print("-" * 60)
    print(
        f"▶ Attack queries in single account: {attack_similar_pairs_s1[0]} similar pairs"
    )
    print(
        f"▶ Attack distributed across accounts: {max(attack_similar_pairs_s2)} pairs max"
    )
    print(
        f"▶ Benign account max similar pairs: {max(benign_similar_pairs) if benign_similar_pairs else 0}"
    )

    print("\n" + "=" * 60)
    print("  DEFENSE EFFECTIVENESS SUMMARY")
    print("=" * 60)

    effectiveness = 0
    if attack_detection_rate > 0:
        effectiveness = 100 - (accounts_needed / 100)  # Higher is better

    if attack_detection_rate >= 0.99 and false_positive_rate <= 0.05:
        status = "EXCELLENT"
    elif attack_detection_rate >= 0.9 and false_positive_rate <= 0.1:
        status = "GOOD"
    elif attack_detection_rate >= 0.7:
        status = "MODERATE"
    else:
        status = "POOR"

    print(f"▶ Detection Status: {status}")
    print(f"▶ Defense Effectiveness Score: {effectiveness:.1f}/100")

    if accounts_needed > 50:
        print("▶ Attack Cost Assessment: HIGH (many accounts needed)")
    elif accounts_needed > 20:
        print("▶ Attack Cost Assessment: MEDIUM (multiple accounts required)")
    else:
        print("▶ Attack Cost Assessment: LOW (few accounts needed)")

    # Save results
    results = {
        "attack_detection_rate": float(attack_detection_rate),
        "false_positive_rate": float(false_positive_rate),
        "similarity_threshold": float(similarity_threshold),
        "pairs_threshold": int(pairs_threshold),
        "accounts_needed": int(accounts_needed),
        "similar_pairs": {
            "scenario1": attack_similar_pairs_s1,
            "scenario2": attack_similar_pairs_s2,
            "scenario3": attack_similar_pairs_s3 if attack_similar_pairs_s3 else [],
            "benign": benign_similar_pairs,
        },
    }

    torch.save(results, os.path.join(args.output_dir, "seat_results.pt"))
    print("\nResults saved to", args.output_dir)
    print("Figures saved to", os.path.join(args.output_dir, "figures"))


if __name__ == "__main__":
    main()
