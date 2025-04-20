import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import PercentFormatter


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize HODA Defense Results")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="defense_results",
        help="Directory containing defense results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="defense_results/visualizations",
        help="Directory to save visualizations",
    )
    return parser.parse_args()


def create_histogram_comparison(metrics, output_dir):
    """
    Create visualization comparing hardness histograms.

    Args:
        metrics: Dictionary of metrics from HODA defense
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get detection metrics
    detection = metrics["detection"]
    threshold = detection["threshold"]
    attack_distance = detection["attack_distance"]
    benign_distance = detection["benign_distance"]
    attack_detected = detection["attack_detected"]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot distance comparison
    labels = ["Benign User", "Attack"]
    distances = [benign_distance, attack_distance]
    colors = ["green", "red"]

    plt.bar(labels, distances, color=colors, alpha=0.7)
    plt.axhline(
        y=threshold, color="black", linestyle="--", label=f"Threshold: {threshold:.4f}"
    )

    plt.ylabel("Pearson Distance")
    plt.title("HODA Detection Results: Hardness-based Distance")
    plt.ylim(0, max(distances) * 1.2)

    # Add value labels
    for i, v in enumerate(distances):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center")

    # Add detection results
    if attack_detected:
        detection_text = "Attack Successfully Detected"
    else:
        detection_text = "Attack Not Detected"

    plt.annotate(
        detection_text,
        xy=(1, attack_distance),
        xytext=(1, attack_distance + max(distances) * 0.3),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.3),
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.savefig(
        os.path.join(output_dir, "distance_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_attack_success_visualization(metrics, output_dir):
    """
    Create visualization of attack success metrics.

    Args:
        metrics: Dictionary of metrics from HODA defense
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get metrics
    original = metrics["original"]
    hr_original = original["hr"]
    hr_surrogate = original["surrogate_hr"]
    attack_success = original["attack_success"] * 100  # Convert to percentage

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot hit rates
    plt.subplot(2, 1, 1)
    models = ["Original Model", "Surrogate Model"]
    hrs = [hr_original, hr_surrogate]
    colors = ["blue", "orange"]

    bars = plt.bar(models, hrs, color=colors, alpha=0.7)
    plt.ylabel("Hit Rate @10")
    plt.title("Model Performance: Original vs. Surrogate")
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.002,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    # Plot attack success rate
    plt.subplot(2, 1, 2)
    plt.bar(["Attack Success Rate"], [attack_success], color="red", alpha=0.7)
    plt.ylabel("Success Rate (%)")
    plt.title("Model Extraction Attack Effectiveness")
    plt.ylim(0, 105)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.grid(True, alpha=0.3)

    # Add value labels
    plt.text(0, attack_success + 2, f"{attack_success:.2f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "attack_success.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_defense_dashboard(metrics, output_dir):
    """
    Create comprehensive defense dashboard.

    Args:
        metrics: Dictionary of metrics from HODA defense
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get metrics
    original = metrics["original"]
    detection = metrics["detection"]

    # Model performance metrics
    hr_original = original["hr"]
    ndcg_original = original["ndcg"]
    hr_surrogate = original["surrogate_hr"]
    ndcg_surrogate = original["surrogate_ndcg"]
    attack_success = original["attack_success"] * 100  # Convert to percentage

    # Detection metrics
    threshold = detection["threshold"]
    attack_distance = detection["attack_distance"]
    benign_distance = detection["benign_distance"]
    attack_detected = detection["attack_detected"]
    false_positive = detection["false_positive"]
    num_queries = detection["num_queries"]

    # Attack metrics from original attack
    attack_metrics = original["attack_metrics"]
    overlap_values = []
    overlap_labels = []

    for key, value in attack_metrics.items():
        if key.startswith("overlap@"):
            k = key.split("@")[1]
            overlap_values.append(value * 100)  # Convert to percentage
            overlap_labels.append(f"Overlap@{k}")

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Title
    plt.suptitle("HODA Defense Evaluation Dashboard", fontsize=16, y=0.98)

    # Plot 1: Model Performance
    ax1 = plt.subplot(2, 2, 1)
    models = ["Original\nHR", "Surrogate\nHR", "Original\nNDCG", "Surrogate\nNDCG"]
    values = [hr_original, hr_surrogate, ndcg_original, ndcg_surrogate]
    colors = ["blue", "orange", "blue", "orange"]

    bars = ax1.bar(models, values, color=colors, alpha=0.7)
    ax1.set_ylabel("Score")
    ax1.set_title("Model Performance")
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.002,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Attack Success Rate
    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(["Attack Success Rate"], [attack_success], color="red", alpha=0.7)
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Model Extraction Attack Effectiveness")
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.grid(True, alpha=0.3)

    # Add value labels
    ax2.text(0, attack_success + 2, f"{attack_success:.2f}%", ha="center", va="bottom")

    # Plot 3: Distance Comparison
    ax3 = plt.subplot(2, 2, 3)
    labels = ["Benign User", "Attack"]
    distances = [benign_distance, attack_distance]
    colors = ["green", "red"]

    ax3.bar(labels, distances, color=colors, alpha=0.7)
    ax3.axhline(
        y=threshold, color="black", linestyle="--", label=f"Threshold: {threshold:.4f}"
    )

    ax3.set_ylabel("Pearson Distance")
    ax3.set_title("HODA Detection Results")
    ax3.set_ylim(0, max(max(distances) * 1.2, threshold * 1.5))
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for i, v in enumerate(distances):
        ax3.text(i, v + 0.02, f"{v:.4f}", ha="center")

    # Plot 4: Overlap Metrics
    ax4 = plt.subplot(2, 2, 4)
    ax4.bar(overlap_labels, overlap_values, color="purple", alpha=0.7)
    ax4.set_ylabel("Overlap (%)")
    ax4.set_title("Attack Overlap Metrics")
    ax4.set_ylim(0, 105)
    ax4.yaxis.set_major_formatter(PercentFormatter())
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for i, v in enumerate(overlap_values):
        ax4.text(i, v + 2, f"{v:.2f}%", ha="center", va="bottom")

    # Add summary text
    summary_text = (
        f"HODA Defense Summary:\n"
        f"• Monitoring {num_queries} queries\n"
        f"• Detection threshold: {threshold:.4f}\n"
        f"• Attack detected: {attack_detected}\n"
        f"• False positive rate: {int(false_positive) * 100}%\n\n"
        f"Attack Effectiveness:\n"
        f"• Attack success rate: {attack_success:.2f}%\n"
        f"• Rank correlation: {attack_metrics.get('rank_correlation', 'N/A'):.4f}"
    )

    fig.text(
        0.5,
        0.02,
        summary_text,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(
        os.path.join(output_dir, "defense_dashboard.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metrics
    metrics_path = os.path.join(args.input_dir, "hoda_metrics.npy")

    try:
        metrics = np.load(metrics_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return

    # Create visualizations
    create_histogram_comparison(metrics, args.output_dir)
    create_attack_success_visualization(metrics, args.output_dir)
    create_defense_dashboard(metrics, args.output_dir)

    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
