import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse


def load_metrics(metrics_file):
    """Load metrics from JSON file"""
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    return metrics


def create_attack_summary(metrics, output_path):
    """Create a comprehensive attack summary visualization"""
    plt.figure(figsize=(12, 8))

    # Create a 2x2 grid for subplots
    gs = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # 1. Overlap Metrics Bar Chart (Top Left)
    ax1 = plt.subplot(gs[0, 0])
    overlap_metrics = {k: v for k, v in metrics.items() if k.startswith("overlap")}
    x = np.arange(len(overlap_metrics))
    bars = ax1.bar(x, list(overlap_metrics.values()), color="cornflowerblue")
    ax1.set_xlabel("Metric")
    ax1.set_ylabel("Score")
    ax1.set_title("Recommendation Overlap")
    ax1.set_xticks(x)
    ax1.set_xticklabels([k.replace("overlap@", "Top-") for k in overlap_metrics.keys()])
    ax1.set_ylim(0, 1.1)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.1%}",
            ha="center",
            va="bottom",
        )

    # 2. Model Fidelity (Top Right)
    ax2 = plt.subplot(gs[0, 1])
    fidelity_data = [metrics.get("rank_correlation", 0), metrics.get("overlap@10", 0)]
    labels = ["Rank Correlation", "Top-10 Overlap"]
    bars = ax2.bar(labels, fidelity_data, color="green")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Model Fidelity Metrics")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.1%}",
            ha="center",
            va="bottom",
        )

    # 3. Overlap by K value (Bottom Left)
    ax3 = plt.subplot(gs[1, 0])
    k_values = [1, 5, 10, 20]
    overlap_by_k = [metrics.get(f"overlap@{k}", 0) for k in k_values]

    bars = ax3.bar(k_values, overlap_by_k, color="orange")
    ax3.set_xlabel("Top-K")
    ax3.set_ylabel("Overlap")
    ax3.set_title("Overlap at Different K Values")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.1%}",
            ha="center",
            va="bottom",
        )

    # 4. Summary box (Bottom Right)
    ax4 = plt.subplot(gs[1, 1])
    ax4.axis("off")

    # Create a gray rectangle
    rect = Rectangle((0.1, 0.1), 0.8, 0.8, facecolor="lightgray", alpha=0.5)
    ax4.add_patch(rect)

    # Add summary text
    summary_text = "ATTACK PERFORMANCE SUMMARY\n"
    summary_text += "-------------------------\n\n"
    summary_text += f"Top-1 Match Rate: {metrics.get('overlap@1', 0):.1%}\n"
    summary_text += f"Top-5 Match Rate: {metrics.get('overlap@5', 0):.1%}\n"
    summary_text += f"Top-10 Match Rate: {metrics.get('overlap@10', 0):.1%}\n"
    summary_text += f"Top-20 Match Rate: {metrics.get('overlap@20', 0):.1%}\n\n"
    summary_text += f"Rank Correlation: {metrics.get('rank_correlation', 0):.2f}\n\n"

    # Calculate attack success metrics
    success_rate = metrics.get("overlap@10", 0) * 100
    success_text = f"Attack Success: "

    if success_rate > 90:
        success_text += "VERY HIGH"
    elif success_rate > 70:
        success_text += "HIGH"
    elif success_rate > 50:
        success_text += "MEDIUM"
    else:
        success_text += "LOW"

    summary_text += success_text

    ax4.text(0.5, 0.5, summary_text, ha="center", va="center", fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Attack summary visualization saved to {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create Attack Summary Visualization")
    parser.add_argument(
        "--metrics",
        type=str,
        default="attack_results/comparison_metrics.json",
        help="Path to metrics JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="attack_results/attack_performance_summary.png",
        help="Output path for summary visualization",
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load metrics
    metrics = load_metrics(args.metrics)

    # Create visualization
    create_attack_summary(metrics, args.output)


if __name__ == "__main__":
    main()
