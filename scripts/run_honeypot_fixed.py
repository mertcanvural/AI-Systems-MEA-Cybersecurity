#!/usr/bin/env python
"""
Run HoneypotNet defense against model extraction attacks.

This script applies the HoneypotNet defense to protect a recommendation model
by injecting a backdoor that transfers to any extracted model, allowing for
both ownership verification and functionality disruption of stolen models.
"""

# Standard library imports
import os
import sys
import argparse
import random

# Third-party imports
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"Added {project_root} to Python path")

# Project imports
try:
    from src.models.base_model import SimpleSequentialRecommender
    from src.data.data_utils import load_movielens, create_train_val_test_splits
    from src.attack.model_extraction import ModelExtractionAttack
    from src.defense.honeypotnet_defense import HoneypotNetDefense

    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    """Run the HoneypotNet defense experiment"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run HoneypotNet defense experiment")

    # Model parameters
    parser.add_argument(
        "--target-model", required=True, help="Path to the target model checkpoint"
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=64, help="Embedding dimension"
    )

    # Data parameters
    parser.add_argument("--data-path", required=True, help="Path to the dataset file")
    parser.add_argument(
        "--min-rating", type=float, default=3.5, help="Minimum rating threshold"
    )

    # Defense parameters
    parser.add_argument(
        "--num-iterations", type=int, default=3, help="Number of attack iterations"
    )
    parser.add_argument(
        "--backdoor-size", type=int, default=10, help="Number of backdoor items"
    )
    parser.add_argument(
        "--trigger-sequence-length",
        type=int,
        default=3,
        help="Length of trigger sequence",
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-defense", action="store_true", help="Save defense artifacts"
    )
    parser.add_argument(
        "--output-dir", default="results/honeypotnet", help="Output directory"
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_path}...")
    data = load_movielens(args.data_path, min_rating=args.min_rating)

    # Get number of items
    num_items = data["num_items"]
    print(f"Dataset contains {num_items} items")

    # Create training/validation/test splits
    print("Creating dataset splits...")
    splits = create_train_val_test_splits(data["user_sequences"])

    print(f"Created {len(splits['train_sequences'])} training sequences")
    print(f"Created {len(splits['val_sequences'])} validation sequences")
    print(f"Created {len(splits['test_sequences'])} test sequences")

    # Extract test sequences
    test_sequences = splits["test_sequences"]
    print(f"Using {len(test_sequences)} test sequences")

    # Load the target model from checkpoint
    print(f"\nLoading target model from {args.target_model}...")
    target_model = SimpleSequentialRecommender(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
    )

    try:
        # Try loading the checkpoint with different formats
        checkpoint = torch.load(args.target_model)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # This is a training checkpoint with extra data
            print("Loading model state from training checkpoint format")
            target_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # This is a direct state dict
            print("Loading model state from state dict format")
            target_model.load_state_dict(checkpoint)

        target_model.eval()
        print("Target model loaded successfully")
    except Exception as e:
        print(f"Error loading target model: {e}")
        sys.exit(1)

    # Create the HoneypotNet defense
    print("\nCreating HoneypotNet defense...")
    defense = HoneypotNetDefense(
        target_model=target_model,
        num_items=num_items,
        backdoor_size=args.backdoor_size,
        trigger_sequence_length=args.trigger_sequence_length,
    )

    # Create the attacker
    attacker = ModelExtractionAttack(
        target_model=target_model,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
    )

    # Run defense experiment
    print("\nRunning defense experiment...")
    metrics = []
    utility_metrics = []

    for iteration in range(1, args.num_iterations + 1):
        print(f"\nIteration {iteration}/{args.num_iterations}")

        # Apply the defense (inject backdoor)
        backdoor_items, trigger_sequence = defense.inject_backdoor()

        # Attack the defended model
        extracted_model = attacker.extract_model(test_sequences)

        # Evaluate backdoor effectiveness
        backdoor_acc = defense.verify_backdoor(extracted_model)
        print(f"Backdoor accuracy: {backdoor_acc:.4f}")

        # Evaluate impact on utility
        utility_impact = defense.evaluate_utility_impact(test_sequences)
        print(f"Utility impact: {utility_impact:.4f}")

        # Record metrics
        metrics.append(
            {
                "iteration": iteration,
                "backdoor_accuracy": backdoor_acc,
                "backdoor_items": backdoor_items,
                "trigger_sequence": trigger_sequence,
            }
        )

        utility_metrics.append(
            {"iteration": iteration, "utility_impact": utility_impact}
        )

    # Save results if requested
    if args.save_defense:
        print("\nSaving defense results...")
        save_results(metrics, utility_metrics, args.output_dir)

        # Create visualizations
        visualize_results(metrics, utility_metrics, args.output_dir)

    print("\nExperiment completed!")


def save_results(metrics, utility_metrics, output_dir):
    """
    Save experiment results to files.

    Args:
        metrics: Attack success metrics
        utility_metrics: Utility preservation metrics
        output_dir: Directory to save results
    """
    # Create results subdirectory
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    results_dir = os.path.join(output_dir, "results")

    # Save metrics to files
    np.save(os.path.join(results_dir, "backdoor_metrics.npy"), metrics)
    np.save(os.path.join(results_dir, "utility_metrics.npy"), utility_metrics)

    # Save summary to text file
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write("HoneypotNet Defense Experiment Summary\n")
        f.write("=====================================\n\n")

        f.write("Backdoor Accuracy:\n")
        for m in metrics:
            f.write(f"Iteration {m['iteration']}: {m['backdoor_accuracy']:.4f}\n")

        f.write("\nUtility Impact:\n")
        for m in utility_metrics:
            f.write(f"Iteration {m['iteration']}: {m['utility_impact']:.4f}\n")


def visualize_results(metrics, utility_metrics, output_dir):
    """
    Create visualizations of the defense effectiveness.

    Args:
        metrics: Attack success metrics
        utility_metrics: Utility preservation metrics
        output_dir: Directory to save visualizations
    """
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")

    # Create a bar plot for the backdoor accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(["Backdoor Accuracy"], [metrics[-1]["backdoor_accuracy"]])
    plt.ylabel("Accuracy")
    plt.title("Backdoor Effectiveness")
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(figures_dir, "backdoor_effectiveness.png"))
    plt.close()

    # Create a line plot for the backdoor accuracy over iterations
    plt.figure(figsize=(10, 6))
    iterations = [m["iteration"] for m in metrics]
    backdoor_accs = [m["backdoor_accuracy"] for m in metrics]
    plt.plot(iterations, backdoor_accs, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Backdoor Accuracy")
    plt.title("Backdoor Effectiveness Over Iterations")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "backdoor_over_iterations.png"))
    plt.close()

    # Create a line plot for the utility impact over iterations
    plt.figure(figsize=(10, 6))
    iterations = [m["iteration"] for m in utility_metrics]
    utility_impacts = [m["utility_impact"] for m in utility_metrics]
    plt.plot(iterations, utility_impacts, marker="o", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Utility Impact")
    plt.title("Utility Impact Over Iterations")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "utility_over_iterations.png"))
    plt.close()


if __name__ == "__main__":
    main()
