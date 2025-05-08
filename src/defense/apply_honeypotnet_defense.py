import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from src.models.base_model import SimpleSequentialRecommender
from src.data.data_utils import load_movielens, create_train_val_test_split
from src.attack.model_extraction import ModelExtractionAttack
from src.defense.honeypotnet_defense import HoneypotNetDefense


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_cc3m_dataset(cc3m_path, num_items, num_samples=5000):
    """
    Load a subset of the CC3M dataset as shadow set for HoneypotNet.
    For simplicity, we simulate loading CC3M by generating random sequences.

    Args:
        cc3m_path: Path to CC3M dataset (not used in this version)
        num_items: Number of items in the dataset
        num_samples: Number of samples to generate

    Returns:
        shadow_set: List of sequences
    """
    # In a real implementation, this would load actual CC3M data
    # For this implementation, we generate random sequences
    shadow_set = []
    for _ in range(num_samples):
        seq_length = np.random.randint(3, 15)
        # Generate random sequence (0 is padding, so we use 1 to num_items)
        sequence = np.random.randint(1, num_items, size=seq_length).tolist()
        shadow_set.append(sequence)

    return shadow_set


def evaluate_defense(target_model, defended_model, test_sequences):
    """
    Evaluate the utility preservation of the defended model.

    Args:
        target_model: Original target model
        defended_model: Model with HoneypotNet defense
        test_sequences: Test sequences for evaluation

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    device = next(target_model.parameters()).device

    # Metrics to evaluate
    ranking_differences = []
    overlap_at_k = {1: [], 5: [], 10: []}

    target_model.eval()
    defended_model.eval()

    with torch.no_grad():
        for sequence in tqdm(test_sequences, desc="Evaluating utility"):
            if not isinstance(sequence, torch.Tensor):
                sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
            else:
                sequence_tensor = sequence.unsqueeze(0).to(device)

            # Get predictions from target model
            target_logits = target_model(sequence_tensor)
            target_scores, target_indices = torch.topk(target_logits[0], k=10)

            # Get predictions from defended model
            defended_logits = defended_model(sequence_tensor)
            defended_scores, defended_indices = torch.topk(defended_logits[0], k=10)

            # Convert to lists
            target_items = target_indices.tolist()
            defended_items = defended_indices.tolist()

            # Calculate overlap at different k values
            for k in overlap_at_k.keys():
                target_at_k = set(target_items[:k])
                defended_at_k = set(defended_items[:k])
                overlap = len(target_at_k.intersection(defended_at_k)) / k
                overlap_at_k[k].append(overlap)

            # Calculate ranking difference
            # Convert to dictionaries mapping item to rank
            target_ranks = {item: i for i, item in enumerate(target_items)}
            defended_ranks = {item: i for i, item in enumerate(defended_items)}

            # Find common items
            common_items = set(target_items).intersection(set(defended_items))

            if common_items:
                # Calculate differences in ranks
                rank_diffs = [
                    abs(target_ranks[item] - defended_ranks[item])
                    for item in common_items
                ]
                ranking_differences.append(np.mean(rank_diffs))

    # Calculate averages
    metrics = {
        "rank_difference": (
            np.mean(ranking_differences) if ranking_differences else float("inf")
        ),
    }

    for k, overlaps in overlap_at_k.items():
        metrics[f"overlap@{k}"] = np.mean(overlaps) if overlaps else 0.0

    return metrics


def evaluate_attack_success(
    defended_model, honeypot_defense, attack_queries=1000, attack_epochs=10
):
    """
    Evaluate how well the defense prevents model extraction attacks.

    Args:
        defended_model: Model with HoneypotNet defense
        honeypot_defense: HoneypotNet defense instance
        attack_queries: Number of queries for the attack
        attack_epochs: Number of training epochs for the surrogate model

    Returns:
        metrics: Dictionary of attack success metrics
    """
    device = next(defended_model.parameters()).device

    # Create attack instance
    attack = ModelExtractionAttack(
        target_model_path="dummy_path",  # Use dummy path as we'll set the model directly
        num_items=defended_model.num_items,
        embedding_dim=defended_model.embedding_dim,
        device=device,
        query_budget=attack_queries,
    )

    # Set the defended model as the target model for the attack
    attack.target_model = defended_model

    # Collect data and train surrogate model
    attack.collect_data(strategy="autoregressive")
    attack.train_surrogate_model(num_epochs=attack_epochs)

    # Generate test sequences for verification
    test_sequences = []
    for _ in range(100):
        seq_length = np.random.randint(3, 10)
        sequence = np.random.randint(
            1, defended_model.num_items, size=seq_length
        ).tolist()
        test_sequences.append(sequence)

    # Verify if the surrogate model has the backdoor
    is_extracted, backdoor_accuracy = honeypot_defense.verify_ownership(
        attack.surrogate_model, test_sequences
    )

    # Evaluate attack effectiveness
    attack_metrics = attack.evaluate_attack(test_sequences=test_sequences)

    # Combine metrics
    metrics = {
        "backdoor_detected": is_extracted,
        "backdoor_accuracy": backdoor_accuracy,
        **attack_metrics,
    }

    return metrics, attack.surrogate_model


def visualize_results(metrics, utility_metrics, output_dir):
    """
    Create visualizations of the defense effectiveness.

    Args:
        metrics: Attack success metrics
        utility_metrics: Utility preservation metrics
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a bar plot for the backdoor accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(["Backdoor Accuracy"], [metrics["backdoor_accuracy"]], color="forestgreen")
    plt.axhline(y=0.1, color="red", linestyle="--", label="Detection Threshold (10%)")
    plt.ylabel("Accuracy")
    plt.title("Backdoor Accuracy in Extracted Model")
    plt.legend()
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(output_dir, "backdoor_accuracy.png"))
    plt.close()

    # Create a bar plot for the overlap metrics
    plt.figure(figsize=(10, 6))
    overlap_values = [metrics[f"overlap@{k}"] for k in [1, 5, 10, 20]]
    plt.bar(["Overlap@1", "Overlap@5", "Overlap@10", "Overlap@20"], overlap_values)
    plt.ylabel("Overlap")
    plt.title("Recommendation Overlap between Target and Extracted Models")
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(output_dir, "recommendation_overlap.png"))
    plt.close()

    # Create a comparison bar plot for utility preservation
    plt.figure(figsize=(10, 6))
    utility_values = [utility_metrics[f"overlap@{k}"] for k in [1, 5, 10]]
    plt.bar(["Overlap@1", "Overlap@5", "Overlap@10"], utility_values)
    plt.ylabel("Overlap")
    plt.title("Utility Preservation: Original vs. Defended Model")
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(output_dir, "utility_preservation.png"))
    plt.close()

    # Save metrics to file
    with open(os.path.join(output_dir, "defense_metrics.txt"), "w") as f:
        f.write("HoneypotNet Defense Metrics\n")
        f.write("=========================\n\n")

        f.write("Attack Prevention Metrics:\n")
        f.write(f"  Backdoor Detected: {metrics['backdoor_detected']}\n")
        f.write(f"  Backdoor Accuracy: {metrics['backdoor_accuracy']:.4f}\n")
        f.write(f"  Rank Correlation: {metrics.get('rank_correlation', 'N/A')}\n")

        for k in [1, 5, 10, 20]:
            if f"overlap@{k}" in metrics:
                f.write(f"  Overlap@{k}: {metrics[f'overlap@{k}']:.4f}\n")

        f.write("\nUtility Preservation Metrics:\n")
        f.write(f"  Rank Difference: {utility_metrics['rank_difference']:.4f}\n")

        for k in [1, 5, 10]:
            f.write(f"  Overlap@{k}: {utility_metrics[f'overlap@{k}']:.4f}\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Apply HoneypotNet Defense to Recommendation Models"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to target model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="defense_results/honeypotnet",
        help="Directory to save defense results",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ml-1m/ratings.dat",
        help="Path to dataset",
    )
    parser.add_argument(
        "--cc3m-path",
        type=str,
        default="data/cc3m",
        help="Path to CC3M dataset (for shadow set)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension for models",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=30,
        help="Number of bi-level optimization iterations",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=5,
        help="Number of epochs for finetuning honeypot layer",
    )
    parser.add_argument(
        "--shadow-train-epochs",
        type=int,
        default=5,
        help="Number of epochs for training shadow model",
    )
    parser.add_argument(
        "--trigger-iterations",
        type=int,
        default=5,
        help="Number of iterations for trigger update",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--lambda-normal",
        type=float,
        default=1.0,
        help="Weight for normal functionality loss",
    )
    parser.add_argument(
        "--lambda-backdoor",
        type=float,
        default=1.0,
        help="Weight for backdoor loss",
    )
    parser.add_argument(
        "--attack-queries",
        type=int,
        default=1000,
        help="Number of queries for model extraction attack evaluation",
    )
    parser.add_argument(
        "--attack-epochs",
        type=int,
        default=10,
        help="Number of epochs for training surrogate models",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    data = load_movielens(args.data_path)
    num_items = data["num_items"]
    print(f"Dataset has {num_items} items")

    # Create train/val/test split
    train_data, val_data, test_data = create_train_val_test_split(data)

    # Load target model
    print(f"Loading target model from {args.target_model}")
    target_model = SimpleSequentialRecommender(num_items, args.embedding_dim)

    try:
        checkpoint = torch.load(args.target_model, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            target_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            target_model.load_state_dict(checkpoint)

        target_model = target_model.to(device)
        target_model.eval()
        print("Target model loaded successfully")
    except Exception as e:
        print(f"Error loading target model: {e}")
        return

    # Load shadow set (CC3M dataset or simulated sequences)
    print(f"Loading shadow set from {args.cc3m_path}")
    shadow_set = load_cc3m_dataset(args.cc3m_path, num_items, num_samples=5000)
    print(f"Shadow set contains {len(shadow_set)} sequences")

    # Initialize HoneypotNet defense
    print("Initializing HoneypotNet defense")
    honeypot_defense = HoneypotNetDefense(
        target_model=target_model,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        device=device,
    )

    # Train the defense
    print("Training HoneypotNet defense")
    honeypot_defense.train(
        shadow_set=shadow_set,
        num_iterations=args.num_iterations,
        finetune_epochs=args.finetune_epochs,
        shadow_train_epochs=args.shadow_train_epochs,
        trigger_iterations=args.trigger_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_normal=args.lambda_normal,
        lambda_backdoor=args.lambda_backdoor,
    )

    # Apply the defense
    print("Applying defense to target model")
    defended_model = honeypot_defense.apply_defense()

    # Extract test sequences from test_data
    test_sequences = []
    for user_id, item_seq in test_data["user_items"].items():
        for i in range(len(item_seq) - 1):
            if i >= 2:  # Use minimum sequence length of 3
                test_sequences.append(item_seq[: i + 1])

    # Select a random subset for evaluation (100 sequences)
    if len(test_sequences) > 100:
        test_sequences = random.sample(test_sequences, 100)

    # Evaluate utility preservation
    print("Evaluating utility preservation")
    utility_metrics = evaluate_defense(target_model, defended_model, test_sequences)

    # Evaluate attack success
    print("Evaluating attack prevention")
    attack_metrics, surrogate_model = evaluate_attack_success(
        defended_model,
        honeypot_defense,
        attack_queries=args.attack_queries,
        attack_epochs=args.attack_epochs,
    )

    # Visualize results
    print("Creating visualizations")
    visualize_results(attack_metrics, utility_metrics, args.output_dir)

    # Save models
    torch.save(
        defended_model.state_dict(), os.path.join(args.output_dir, "defended_model.pt")
    )
    torch.save(
        surrogate_model.state_dict(),
        os.path.join(args.output_dir, "surrogate_model.pt"),
    )

    # Save defense
    honeypot_defense.save(os.path.join(args.output_dir, "honeypot_defense.pt"))

    print(f"Defense application complete. Results saved to {args.output_dir}")
    print("\nKey metrics:")
    print(f"  Utility Preservation (Overlap@10): {utility_metrics['overlap@10']:.4f}")
    print(f"  Backdoor Detection: {attack_metrics['backdoor_detected']}")
    print(f"  Backdoor Accuracy: {attack_metrics['backdoor_accuracy']:.4f}")
    print(f"  Attack Success (Overlap@10): {attack_metrics['overlap@10']:.4f}")


if __name__ == "__main__":
    main()
