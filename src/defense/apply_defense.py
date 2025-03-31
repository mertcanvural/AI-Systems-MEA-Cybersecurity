import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import SimpleSequentialRecommender
from src.defense.gro_defense import GRODefense
from src.attack.model_extraction import ModelExtractionAttack
from src.data.data_utils import load_movielens


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Apply Defense Against Model Extraction Attacks"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to target model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ml-1m/ratings.dat",
        help="Path to dataset",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension for models",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of epochs for defense training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lambda-swap",
        type=float,
        default=0.1,
        help="Weight of the swap loss",
    )
    parser.add_argument(
        "--attack-queries",
        type=int,
        default=1000,
        help="Number of queries for model extraction attack",
    )
    parser.add_argument(
        "--attack-epochs",
        type=int,
        default=10,
        help="Number of epochs for training surrogate models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="defense_results",
        help="Directory to save results",
    )
    return parser.parse_args()


def prepare_data(data_path, batch_size):
    """Prepare data for training and evaluation"""
    print("Loading and preparing data...")

    # Load data
    data = load_movielens(data_path)
    num_items = data["num_items"]

    # Create datasets
    train_seqs = []
    train_targets = []
    val_seqs = []
    val_targets = []
    test_seqs = []
    test_targets = []

    for user_id, sequence in data["user_sequences"].items():
        if len(sequence) < 3:
            continue

        # Last item is test, second-to-last is validation
        test_item = sequence[-1]
        val_item = sequence[-2]
        train_seq = sequence[:-2]

        # Skip if train sequence is empty
        if len(train_seq) == 0:
            continue

        train_seqs.append(train_seq)
        train_targets.append(val_item)

        val_seqs.append(sequence[:-1])
        val_targets.append(test_item)

        test_seqs.append(sequence)
        test_targets.append(0)  # Placeholder, not used

    # Create data loaders
    def pad_sequences(sequences):
        max_len = max(len(seq) for seq in sequences)
        padded = np.zeros((len(sequences), max_len), dtype=np.int64)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq
        return padded

    train_seqs_padded = pad_sequences(train_seqs)
    train_targets = np.array(train_targets, dtype=np.int64)

    val_seqs_padded = pad_sequences(val_seqs)
    val_targets = np.array(val_targets, dtype=np.int64)

    test_seqs_padded = pad_sequences(test_seqs)
    test_targets = np.array(test_targets, dtype=np.int64)

    train_dataset = TensorDataset(
        torch.LongTensor(train_seqs_padded), torch.LongTensor(train_targets)
    )

    val_dataset = TensorDataset(
        torch.LongTensor(val_seqs_padded), torch.LongTensor(val_targets)
    )

    test_dataset = TensorDataset(
        torch.LongTensor(test_seqs_padded), torch.LongTensor(test_targets)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset has {num_items} items")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_items": num_items,
    }


def load_target_model(model_path, num_items, embedding_dim, device):
    """Load target model from checkpoint"""
    model = SimpleSequentialRecommender(num_items, embedding_dim)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded target model from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}"
            )
        else:
            model.load_state_dict(checkpoint)
            print("Loaded target model state dict")

        model = model.to(device)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading target model: {e}")
        raise e


def evaluate_model(model, data_loader, device, top_k=10):
    """Evaluate model performance"""
    model.eval()

    hits = 0
    ndcg = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            sequences, targets = batch

            sequences = sequences.to(device)
            targets = targets.to(device)

            # Get predictions
            logits = model(sequences)

            # Get top-k recommendations
            _, indices = torch.topk(logits, k=top_k, dim=1)

            # Calculate metrics
            for i, target in enumerate(targets):
                total += 1
                if target.item() in indices[i]:
                    hits += 1
                    rank = torch.where(indices[i] == target.item())[0][0].item() + 1
                    ndcg += 1.0 / np.log2(rank + 1)

    hr = hits / total if total > 0 else 0
    ndcg = ndcg / total if total > 0 else 0

    return hr, ndcg


def attack_model(
    target_model, num_items, embedding_dim, device, query_budget=1000, attack_epochs=10
):
    """Apply model extraction attack on the target model"""
    print(f"Applying model extraction attack with {query_budget} queries...")

    # Create a temporary file path but don't use it
    # The target model will be set directly instead
    attack = ModelExtractionAttack(
        # Pass a dummy path that won't be used
        target_model_path="dummy_path",
        num_items=num_items,
        embedding_dim=embedding_dim,
        device=device,
        query_budget=query_budget,
        top_k=10,
        # Pass the model directly as surrogate_model to avoid loading from path
        surrogate_model=SimpleSequentialRecommender(num_items, embedding_dim),
    )

    # Set the target model directly - this overrides the _load_target_model call
    attack.target_model = target_model

    # Collect data through queries
    attack.collect_data(strategy="autoregressive")

    # Train surrogate model
    attack.train_surrogate_model(num_epochs=attack_epochs)

    # Evaluate attack
    attack_metrics = attack.evaluate_attack()

    return attack.surrogate_model, attack_metrics


def apply_defense_and_evaluate(args):
    """Apply GRO defense and evaluate its effectiveness"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    data = prepare_data(args.data_path, args.batch_size)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    num_items = data["num_items"]

    # Load target model
    target_model = load_target_model(
        args.target_model, num_items, args.embedding_dim, device
    )

    # Evaluate original model
    print("Evaluating original model...")
    original_hr, original_ndcg = evaluate_model(target_model, val_loader, device)
    print(f"Original model - HR@10: {original_hr:.4f}, NDCG@10: {original_ndcg:.4f}")

    # Attack original model
    print("Attacking original model...")
    original_surrogate, original_attack_metrics = attack_model(
        target_model,
        num_items,
        args.embedding_dim,
        device,
        args.attack_queries,
        args.attack_epochs,
    )

    # Evaluate surrogate model from attack on original model
    print("Evaluating surrogate model (attack on original)...")
    original_surrogate_hr, original_surrogate_ndcg = evaluate_model(
        original_surrogate, val_loader, device
    )
    print(
        f"Surrogate of original - HR@10: {original_surrogate_hr:.4f}, NDCG@10: {original_surrogate_ndcg:.4f}"
    )

    # Apply GRO defense
    print("Applying GRO defense to the model...")
    defense = GRODefense(
        target_model=target_model,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        lambda_swap=args.lambda_swap,
        device=device,
    )

    # Train the model with defense
    defense.train(train_loader, val_loader, num_epochs=args.num_epochs)

    # Save the defended model
    os.makedirs(args.output_dir, exist_ok=True)
    defended_model_path = os.path.join(args.output_dir, "defended_model.pt")
    defense.save_model(defended_model_path)

    # Evaluate defended model
    print("Evaluating defended model...")
    defended_model = defense.target_model
    defended_hr, defended_ndcg = evaluate_model(defended_model, val_loader, device)
    print(f"Defended model - HR@10: {defended_hr:.4f}, NDCG@10: {defended_ndcg:.4f}")

    # Attack defended model
    print("Attacking defended model...")
    defended_surrogate, defended_attack_metrics = attack_model(
        defended_model,
        num_items,
        args.embedding_dim,
        device,
        args.attack_queries,
        args.attack_epochs,
    )

    # Evaluate surrogate model from attack on defended model
    print("Evaluating surrogate model (attack on defended)...")
    defended_surrogate_hr, defended_surrogate_ndcg = evaluate_model(
        defended_surrogate, val_loader, device
    )
    print(
        f"Surrogate of defended - HR@10: {defended_surrogate_hr:.4f}, NDCG@10: {defended_surrogate_ndcg:.4f}"
    )

    # Save surrogate models
    torch.save(
        original_surrogate.state_dict(),
        os.path.join(args.output_dir, "original_surrogate.pt"),
    )
    torch.save(
        defended_surrogate.state_dict(),
        os.path.join(args.output_dir, "defended_surrogate.pt"),
    )

    # Compute metrics to measure defense effectiveness
    original_attack_success = (
        original_surrogate_hr / original_hr if original_hr > 0 else 0
    )
    defended_attack_success = (
        defended_surrogate_hr / defended_hr if defended_hr > 0 else 0
    )

    utility_preservation = defended_hr / original_hr if original_hr > 0 else 0
    defense_effectiveness = (
        1 - (defended_attack_success / original_attack_success)
        if original_attack_success > 0
        else 0
    )

    # Print summary
    print("\n" + "=" * 50)
    print("DEFENSE EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Original model HR@10: {original_hr:.4f}, NDCG@10: {original_ndcg:.4f}")
    print(f"Defended model HR@10: {defended_hr:.4f}, NDCG@10: {defended_ndcg:.4f}")
    print(f"Utility preservation: {utility_preservation:.2%}")
    print("-" * 50)
    print(
        f"Surrogate of original HR@10: {original_surrogate_hr:.4f}, NDCG@10: {original_surrogate_ndcg:.4f}"
    )
    print(
        f"Surrogate of defended HR@10: {defended_surrogate_hr:.4f}, NDCG@10: {defended_surrogate_ndcg:.4f}"
    )
    print(f"Original attack success rate: {original_attack_success:.2%}")
    print(f"Defended attack success rate: {defended_attack_success:.2%}")
    print(f"Defense effectiveness: {defense_effectiveness:.2%}")
    print("=" * 50)

    # Save metrics to file
    metrics = {
        "original": {
            "hr": original_hr,
            "ndcg": original_ndcg,
            "surrogate_hr": original_surrogate_hr,
            "surrogate_ndcg": original_surrogate_ndcg,
            "attack_success": original_attack_success,
            "attack_metrics": original_attack_metrics,
        },
        "defended": {
            "hr": defended_hr,
            "ndcg": defended_ndcg,
            "surrogate_hr": defended_surrogate_hr,
            "surrogate_ndcg": defended_surrogate_ndcg,
            "attack_success": defended_attack_success,
            "attack_metrics": defended_attack_metrics,
        },
        "summary": {
            "utility_preservation": utility_preservation,
            "defense_effectiveness": defense_effectiveness,
        },
    }

    import json

    with open(os.path.join(args.output_dir, "defense_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Create visualization of defense results
    create_defense_visualization(metrics, args.output_dir)

    print(f"Results saved to {args.output_dir}")

    return metrics


def create_defense_visualization(metrics, output_dir):
    """Create visualization of defense results"""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create directory for figures
    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics
    original_hr = metrics["original"]["hr"]
    original_surrogate_hr = metrics["original"]["surrogate_hr"]
    defended_hr = metrics["defended"]["hr"]
    defended_surrogate_hr = metrics["defended"]["surrogate_hr"]

    original_ndcg = metrics["original"]["ndcg"]
    original_surrogate_ndcg = metrics["defended"]["ndcg"]
    defended_ndcg = metrics["defended"]["ndcg"]
    defended_surrogate_ndcg = metrics["defended"]["surrogate_ndcg"]

    original_attack_success = metrics["original"]["attack_success"]
    defended_attack_success = metrics["defended"]["attack_success"]
    utility_preservation = metrics["summary"]["utility_preservation"]
    defense_effectiveness = metrics["summary"]["defense_effectiveness"]

    # Improved HR comparison with normalized values and relative performance
    plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])

    # 1. Top-left: HR Comparison Bar Chart
    ax1 = plt.subplot(gs[0, 0])
    bar_width = 0.35
    x = np.array([0, 1])

    # Calculate relative performance vs. original model (set to 100%)
    rel_original = 100.0  # original model is the baseline (100%)
    rel_original_surrogate = (original_surrogate_hr / original_hr) * 100.0
    rel_defended = (defended_hr / original_hr) * 100.0
    rel_defended_surrogate = (defended_surrogate_hr / original_hr) * 100.0

    # Original model group
    ax1.bar(0, rel_original, width=bar_width, color="blue", alpha=0.7, label="Original")
    ax1.bar(
        0 + bar_width,
        rel_original_surrogate,
        width=bar_width,
        color="orange",
        alpha=0.7,
        label="Surrogate (Attack on Original)",
    )

    # Defended model group
    ax1.bar(
        1.5, rel_defended, width=bar_width, color="green", alpha=0.7, label="Defended"
    )
    ax1.bar(
        1.5 + bar_width,
        rel_defended_surrogate,
        width=bar_width,
        color="red",
        alpha=0.7,
        label="Surrogate (Attack on Defended)",
    )

    # Add absolute value labels
    def add_value_labels(bars, values, format_str="{:.4f}"):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 2,
                format_str.format(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_value_labels([ax1.patches[0], ax1.patches[2]], [original_hr, defended_hr])
    add_value_labels(
        [ax1.patches[1], ax1.patches[3]], [original_surrogate_hr, defended_surrogate_hr]
    )

    ax1.set_ylabel("Performance Relative to Original Model (%)")
    ax1.set_title("Recommendation Performance Impact", fontsize=12, fontweight="bold")
    ax1.set_xticks([0 + bar_width / 2, 1.5 + bar_width / 2])
    ax1.set_xticklabels(["Original Model Group", "Defended Model Group"])
    ax1.set_ylim(
        0,
        max(rel_original, rel_original_surrogate, rel_defended, rel_defended_surrogate)
        * 1.15,
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right")

    # 2. Top-right: Attack Success Rate Comparison
    ax2 = plt.subplot(gs[0, 1])

    # For better visualization, compute how well the surrogate model copies the original
    original_copying_ability = (original_surrogate_hr / original_hr) * 100
    defended_copying_ability = (defended_surrogate_hr / defended_hr) * 100

    # Also calculate the absolute performance drop from defense
    abs_performance_drop = original_hr - defended_hr
    rel_performance_drop = (1 - (defended_hr / original_hr)) * 100

    # Calculate the attack effectiveness drop
    attack_effectiveness_drop = original_copying_ability - defended_copying_ability

    values = [
        original_copying_ability,
        defended_copying_ability,
        attack_effectiveness_drop,
    ]
    bars = ax2.bar(
        [
            "Original Model\nAttack Success",
            "Defended Model\nAttack Success",
            "Attack\nEffectiveness Drop",
        ],
        values,
        color=["orange", "red", "green"],
        alpha=0.7,
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Attack Success Rate Comparison", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_ylim(0, max(original_copying_ability, defended_copying_ability) * 1.15)

    # 3. Bottom-left: Defense Impact Metrics
    ax3 = plt.subplot(gs[1, 0])

    # Extract overlap metrics for visualization
    overlap_metrics = []
    original_overlaps = []
    defended_overlaps = []

    for k in [1, 5, 10, 20]:
        key = f"overlap@{k}"
        if (
            key in metrics["original"]["attack_metrics"]
            and key in metrics["defended"]["attack_metrics"]
        ):
            overlap_metrics.append(f"Overlap@{k}")
            original_overlaps.append(metrics["original"]["attack_metrics"][key])
            defended_overlaps.append(metrics["defended"]["attack_metrics"][key])

    x = np.arange(len(overlap_metrics))
    width = 0.35

    ax3.bar(
        x - width / 2,
        original_overlaps,
        width,
        label="Original",
        color="blue",
        alpha=0.7,
    )
    ax3.bar(
        x + width / 2,
        defended_overlaps,
        width,
        label="Defended",
        color="green",
        alpha=0.7,
    )

    # Add reduction percentage
    for i in range(len(overlap_metrics)):
        orig = original_overlaps[i]
        def_val = defended_overlaps[i]
        reduction = (orig - def_val) / orig * 100 if orig > 0 else 0

        if reduction > 0:
            ax3.text(
                i,
                min(orig, def_val) - 0.05,
                f"↓{reduction:.1f}%",
                ha="center",
                va="top",
                fontweight="bold",
                color="green",
            )

    ax3.set_xlabel("Metric")
    ax3.set_ylabel("Overlap Score")
    ax3.set_title("Attack Overlap Metrics", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(overlap_metrics)
    ax3.legend()
    ax3.grid(axis="y", linestyle="--", alpha=0.3)

    # 4. Bottom-right: Impact Summary
    ax4 = plt.subplot(gs[1, 1])
    ax4.axis("off")

    # Create a summary table with key metrics
    orig_rank_corr = metrics["original"]["attack_metrics"].get("rank_correlation", 0)
    def_rank_corr = metrics["defended"]["attack_metrics"].get("rank_correlation", 0)

    table_data = [
        ["Metric", "Original", "Defended", "Change"],
        [
            "Utility (HR@10)",
            f"{original_hr:.4f}",
            f"{defended_hr:.4f}",
            f"{utility_preservation:.1f}%",
        ],
        [
            "Attack Success",
            f"{original_attack_success:.1%}",
            f"{defended_attack_success:.1%}",
            f"{(defended_attack_success-original_attack_success)*100:.1f}%",
        ],
        [
            "Rank Correlation",
            f"{orig_rank_corr:.3f}",
            f"{def_rank_corr:.3f}",
            f"{def_rank_corr-orig_rank_corr:.3f}",
        ],
        [
            "Overlap@10",
            f"{original_overlaps[2]:.2f}",
            f"{defended_overlaps[2]:.2f}",
            f"{((defended_overlaps[2]-original_overlaps[2])/original_overlaps[2]*100):.1f}%",
        ],
    ]

    table = ax4.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        cellColours=[["#f0f0f0"] * 4]
        + [
            ["#ffffff", "#d6eaf8", "#d5f5e3", "#fcf3cf" if i == 1 else "#d5f5e3"]
            for i in range(4)
        ],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Add title above table
    ax4.text(
        0.5,
        0.9,
        "Defense Impact Summary",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Add interpretation text
    if attack_effectiveness_drop > 0:
        effectiveness_text = (
            "✓ Defense successfully reduced attack effectiveness\n"
            f"✓ Attack overlap@10 reduced by {(original_overlaps[2]-defended_overlaps[2])/original_overlaps[2]*100:.1f}%\n"
            f"• Performance trade-off: {rel_performance_drop:.1f}% HR reduction"
        )
        color = "green"
    else:
        effectiveness_text = (
            "✗ Defense did not reduce attack effectiveness\n"
            "✗ Surrogate model performance remains high\n"
            "• Consider increasing defense strength"
        )
        color = "red"

    ax4.text(
        0.5,
        0.1,
        effectiveness_text,
        ha="center",
        va="center",
        fontsize=10,
        color=color,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comprehensive_defense_analysis.png"), dpi=300)
    plt.close()

    # Create additional focused visualizations

    # 1. HR Comparison (Original format but with clearer labeling)
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.array([0, 1])
    original_bars = plt.bar(
        x,
        [original_hr, original_surrogate_hr],
        width=bar_width,
        label="Original",
        color="blue",
        alpha=0.7,
    )
    defended_bars = plt.bar(
        x + bar_width,
        [defended_hr, defended_surrogate_hr],
        width=bar_width,
        label="Defended",
        color="green",
        alpha=0.7,
    )

    plt.xlabel("Model")
    plt.ylabel("Hit Ratio @10")
    plt.title("Hit Ratio Comparison: Original vs. Defended Models")
    plt.xticks(x + bar_width / 2, ["Target Model", "Surrogate Model"])
    plt.legend()

    # Add value labels
    for bar in original_bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    for bar in defended_bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hr_comparison.png"))
    plt.close()

    # 2. Defense summary metrics but improved
    plt.figure(figsize=(10, 6))
    metrics_names = [
        "Utility\nPreservation",
        "Original Attack\nSuccess Rate",
        "Defended Attack\nSuccess Rate",
        "Attack\nEffectiveness\nDrop",
    ]
    metrics_values = [
        utility_preservation / 100,
        original_attack_success,
        defended_attack_success,
        attack_effectiveness_drop / 100,
    ]

    colors = ["blue", "orange", "red", "green"]
    bars = plt.bar(metrics_names, metrics_values, color=colors)

    plt.ylabel("Rate")
    plt.title("Defense Performance Metrics")
    plt.ylim(0, 1.1)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{height:.2%}" if bar.get_x() != bars[-1].get_x() else f"{height:.1%}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "defense_summary.png"))
    plt.close()

    # 3. Overlap comparison with clearer reduction indicators
    plt.figure(figsize=(12, 6))
    x = np.arange(len(overlap_metrics))
    bar_width = 0.35

    original_bars = plt.bar(
        x,
        original_overlaps,
        width=bar_width,
        label="Original Model",
        color="blue",
        alpha=0.7,
    )
    defended_bars = plt.bar(
        x + bar_width,
        defended_overlaps,
        width=bar_width,
        label="Defended Model",
        color="green",
        alpha=0.7,
    )

    plt.xlabel("Metric")
    plt.ylabel("Overlap")
    plt.title("Attack Performance Metrics: Original vs. Defended")
    plt.xticks(x + bar_width / 2, overlap_metrics)
    plt.legend()

    # Add value labels with reduction indicators
    for i, (o_bar, d_bar) in enumerate(zip(original_bars, defended_bars)):
        o_height = o_bar.get_height()
        d_height = d_bar.get_height()

        plt.text(
            o_bar.get_x() + o_bar.get_width() / 2.0,
            o_height + 0.02,
            f"{o_height:.2f}",
            ha="center",
            va="bottom",
        )

        # Add percentage reduction
        reduction = (o_height - d_height) / o_height * 100 if o_height > 0 else 0
        plt.text(
            d_bar.get_x() + d_bar.get_width() / 2.0,
            d_height + 0.02,
            f"{d_height:.2f}\n↓{reduction:.1f}%",
            ha="center",
            va="bottom",
            color="green" if reduction > 0 else "black",
        )

    # Add rank correlation
    plt.figtext(
        0.5,
        0.01,
        f"Rank Correlation - Original: {orig_rank_corr:.3f}, Defended: {def_rank_corr:.3f}",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5},
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(output_dir, "attack_metrics_comparison.png"))
    plt.close()


def main():
    """Main function"""
    args = parse_args()
    apply_defense_and_evaluate(args)


if __name__ == "__main__":
    main()
