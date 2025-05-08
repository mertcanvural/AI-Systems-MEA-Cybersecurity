#!/usr/bin/env python
"""
Generate visualization figures for the deception defense experiments.

This script creates visualizations of the HoneypotNet defense performance,
showing the trade-off between utility preservation and defense effectiveness.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the visualization style
sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)


def create_defense_comparison():
    """Create a comparison of different defense strategies."""
    # Sample data (in a real scenario, this would be loaded from experiment results)
    defenses = [
        "No Defense",
        "Query Rate Limiting",
        "Model Perturbation",
        "HoneypotNet",
    ]

    # Effectiveness measures against model extraction
    extraction_resistance = [0.0, 0.4, 0.65, 0.85]  # Higher is better

    # Utility preservation (original model performance maintained)
    utility_preservation = [1.0, 0.95, 0.80, 0.92]  # Higher is better

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set width of bars
    barWidth = 0.35

    # Set positions of the bars on X axis
    r1 = np.arange(len(defenses))
    r2 = [x + barWidth for x in r1]

    # Create bars
    ax.bar(
        r1,
        extraction_resistance,
        width=barWidth,
        label="Extraction Resistance",
        color="royalblue",
        edgecolor="grey",
    )
    ax.bar(
        r2,
        utility_preservation,
        width=barWidth,
        label="Utility Preservation",
        color="lightcoral",
        edgecolor="grey",
    )

    # Add labels and legend
    ax.set_xlabel("Defense Strategy")
    ax.set_ylabel("Effectiveness Score (0-1)")
    ax.set_title("Comparison of Model Extraction Defense Strategies")
    ax.set_xticks([r + barWidth / 2 for r in range(len(defenses))])
    ax.set_xticklabels(defenses)
    ax.set_ylim(0, 1.1)

    # Add value labels on the bars
    for i, v in enumerate(extraction_resistance):
        ax.text(r1[i], v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    for i, v in enumerate(utility_preservation):
        ax.text(r2[i], v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    ax.legend(loc="lower right")

    # Save the figure
    os.makedirs("figures/deception", exist_ok=True)
    plt.savefig(
        "figures/deception/defense_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("✅ Created defense comparison figure")


def create_honeypotnet_architecture():
    """Create a visualization of the HoneypotNet architecture."""
    # Create a new figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Hide axes
    ax.axis("off")

    # Define component sizes and positions
    component_height = 0.15
    component_width = 0.3

    # Colors
    colors = {
        "model": "royalblue",
        "honeypot": "firebrick",
        "trigger": "forestgreen",
        "normal": "lightgray",
        "extracted": "orange",
        "arrow": "black",
    }

    # Draw the original model
    ax.add_patch(
        plt.Rectangle(
            (0.1, 0.7),
            component_width,
            component_height,
            facecolor=colors["model"],
            alpha=0.8,
            label="Original Model",
        )
    )
    ax.text(
        0.1 + component_width / 2,
        0.7 + component_height / 2,
        "Original Model",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )

    # Draw the honeypot layer
    ax.add_patch(
        plt.Rectangle(
            (0.1, 0.5),
            component_width,
            component_height,
            facecolor=colors["honeypot"],
            alpha=0.8,
            label="Honeypot Layer",
        )
    )
    ax.text(
        0.1 + component_width / 2,
        0.5 + component_height / 2,
        "Honeypot Layer",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )

    # Draw the combined defended model
    ax.add_patch(
        plt.Rectangle(
            (0.5, 0.6),
            component_width,
            component_height * 2,
            facecolor=colors["model"],
            alpha=0.8,
            label="Defended Model",
        )
    )
    ax.text(
        0.5 + component_width / 2,
        0.6 + component_height,
        "Defended Model",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )

    # Draw the trigger sequence
    ax.add_patch(
        plt.Rectangle(
            (0.3, 0.3),
            component_width * 0.5,
            component_height,
            facecolor=colors["trigger"],
            alpha=0.8,
            label="Trigger Sequence",
        )
    )
    ax.text(
        0.3 + component_width * 0.25,
        0.3 + component_height / 2,
        "Trigger",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )

    # Draw the normal query
    ax.add_patch(
        plt.Rectangle(
            (0.1, 0.3),
            component_width * 0.5,
            component_height,
            facecolor=colors["normal"],
            alpha=0.8,
            label="Normal Query",
        )
    )
    ax.text(
        0.1 + component_width * 0.25,
        0.3 + component_height / 2,
        "Normal\nQuery",
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Draw the extracted model
    ax.add_patch(
        plt.Rectangle(
            (0.5, 0.3),
            component_width,
            component_height,
            facecolor=colors["extracted"],
            alpha=0.8,
            label="Extracted Model",
        )
    )
    ax.text(
        0.5 + component_width / 2,
        0.3 + component_height / 2,
        "Extracted Model",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )

    # Draw the backdoor activation
    ax.add_patch(
        plt.Rectangle(
            (0.5, 0.1),
            component_width,
            component_height,
            facecolor="red",
            alpha=0.8,
            label="Backdoor Activated",
        )
    )
    ax.text(
        0.5 + component_width / 2,
        0.1 + component_height / 2,
        "Backdoor Activated",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )

    # Draw arrows
    arrow_props = dict(arrowstyle="->", lw=2, color=colors["arrow"])

    # Original + Honeypot -> Defended
    ax.annotate("", xy=(0.5, 0.7), xytext=(0.4, 0.7), arrowprops=arrow_props)
    ax.annotate("", xy=(0.5, 0.6), xytext=(0.4, 0.55), arrowprops=arrow_props)

    # Normal query -> Defended model
    ax.annotate("", xy=(0.5, 0.65), xytext=(0.3, 0.35), arrowprops=arrow_props)
    ax.text(0.35, 0.45, "Normal\nResponse", ha="center", va="center", fontsize=10)

    # Trigger -> Defended model
    ax.annotate("", xy=(0.65, 0.6), xytext=(0.55, 0.35), arrowprops=arrow_props)
    ax.text(
        0.65,
        0.45,
        "Backdoor\nResponse",
        ha="center",
        va="center",
        fontsize=10,
        color="red",
    )

    # Attacker queries -> Extracted model
    ax.annotate("", xy=(0.65, 0.3), xytext=(0.8, 0.5), arrowprops=arrow_props)
    ax.text(0.75, 0.45, "Model\nExtraction", ha="center", va="center", fontsize=10)

    # Trigger -> Extracted model
    ax.annotate("", xy=(0.6, 0.3), xytext=(0.55, 0.3), arrowprops=arrow_props)

    # Extracted model -> Backdoor activation
    ax.annotate("", xy=(0.65, 0.1), xytext=(0.65, 0.3), arrowprops=arrow_props)

    # Add title
    ax.set_title("HoneypotNet Defense Mechanism Architecture", fontsize=18, y=0.98)

    # Add legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["model"], alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["honeypot"], alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["trigger"], alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["extracted"], alpha=0.8),
    ]
    labels = ["Original Model", "Honeypot Layer", "Trigger Sequence", "Extracted Model"]
    ax.legend(handles, labels, loc="upper right", fontsize=12)

    # Save the figure
    os.makedirs("figures/deception", exist_ok=True)
    plt.savefig(
        "figures/deception/honeypotnet_architecture.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("✅ Created HoneypotNet architecture figure")


def create_performance_tradeoff():
    """Create a visualization of the utility-protection trade-off."""
    # Sample data (in a real scenario, this would be loaded from experiment results)
    backdoor_strength = np.linspace(0, 1, 10)  # Backdoor strength from 0 to 1

    # Utility is high when backdoor is weak, and decreases as backdoor gets stronger
    utility = 1 - backdoor_strength * 0.3 - 0.1 * np.random.rand(10)
    utility = np.clip(utility, 0, 1)

    # Protection is low when backdoor is weak, and increases as backdoor gets stronger
    protection = backdoor_strength**0.8 + 0.1 * np.random.rand(10)
    protection = np.clip(protection, 0, 1)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the trade-off curves
    ax.plot(
        backdoor_strength,
        utility,
        "o-",
        color="royalblue",
        linewidth=2,
        label="Utility Preservation",
    )
    ax.plot(
        backdoor_strength,
        protection,
        "o-",
        color="firebrick",
        linewidth=2,
        label="Protection Effectiveness",
    )

    # Plot the balanced point (the intersection)
    intersection_idx = np.argmin(np.abs(utility - protection))
    ax.plot(
        backdoor_strength[intersection_idx],
        utility[intersection_idx],
        "o",
        color="forestgreen",
        markersize=10,
        label=f"Balanced Point ({backdoor_strength[intersection_idx]:.2f})",
    )

    # Add a vertical line at the balanced point
    ax.axvline(
        x=backdoor_strength[intersection_idx], color="gray", linestyle="--", alpha=0.5
    )

    # Add shaded regions for recommended operation zones
    ax.axvspan(
        backdoor_strength[intersection_idx] - 0.1,
        backdoor_strength[intersection_idx] + 0.1,
        alpha=0.2,
        color="green",
        label="Recommended Zone",
    )

    # Add labels and legend
    ax.set_xlabel("Backdoor Strength")
    ax.set_ylabel("Effectiveness Score (0-1)")
    ax.set_title("Trade-off between Utility Preservation and Protection Effectiveness")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")

    # Add annotations
    ax.annotate(
        "Higher Utility",
        xy=(0.1, 0.9),
        xytext=(0.2, 0.95),
        arrowprops=dict(arrowstyle="->"),
    )
    ax.annotate(
        "Higher Protection",
        xy=(0.9, 0.9),
        xytext=(0.8, 0.95),
        arrowprops=dict(arrowstyle="->"),
    )

    # Save the figure
    os.makedirs("figures/deception", exist_ok=True)
    plt.savefig(
        "figures/deception/performance_tradeoff.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("✅ Created performance trade-off figure")


def create_attack_success_comparison():
    """Create a visualization comparing attack success with and without defense."""
    # Sample data (in a real scenario, this would be loaded from experiment results)
    query_numbers = [100, 500, 1000, 2000, 5000, 10000]

    # Attack success rate without defense
    undefended_success = [0.2, 0.45, 0.6, 0.75, 0.85, 0.9]

    # Attack success rate with HoneypotNet defense
    defended_success = [0.15, 0.3, 0.4, 0.45, 0.5, 0.52]

    # Backdoor detection rate
    backdoor_detection = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Create the figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot attack success rates
    ax1.plot(
        query_numbers,
        undefended_success,
        "o-",
        color="firebrick",
        linewidth=2,
        label="Undefended Model",
    )
    ax1.plot(
        query_numbers,
        defended_success,
        "o-",
        color="royalblue",
        linewidth=2,
        label="HoneypotNet Defended",
    )

    # Set labels for first axis
    ax1.set_xlabel("Number of Queries")
    ax1.set_ylabel("Attack Success Rate", color="black")
    ax1.tick_params(axis="y")
    ax1.set_ylim(0, 1)

    # Create second y-axis for backdoor detection rate
    ax2 = ax1.twinx()
    ax2.plot(
        query_numbers,
        backdoor_detection,
        "o--",
        color="forestgreen",
        linewidth=2,
        label="Backdoor Detection Rate",
    )
    ax2.set_ylabel("Detection Rate", color="forestgreen")
    ax2.tick_params(axis="y", colors="forestgreen")
    ax2.set_ylim(0, 1)

    # Add title
    ax1.set_title("Model Extraction Attack Success vs. Defense Effectiveness")

    # Add grid
    ax1.grid(True, alpha=0.3)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # Add annotations
    ax1.annotate(
        "Protection Gap",
        xy=(5000, 0.4),
        xytext=(3000, 0.3),
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
    )

    # Log scale for x-axis
    ax1.set_xscale("log")

    # Save the figure
    os.makedirs("figures/deception", exist_ok=True)
    plt.savefig(
        "figures/deception/attack_success_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("✅ Created attack success comparison figure")


def main():
    """Generate all figures for the deception defense."""
    print("Generating deception defense visualization figures...")

    # Create the figures directory if it doesn't exist
    os.makedirs("figures/deception", exist_ok=True)

    # Generate figures
    create_defense_comparison()
    create_honeypotnet_architecture()
    create_performance_tradeoff()
    create_attack_success_comparison()

    print("\nAll deception defense figures have been generated in 'figures/deception/'")


if __name__ == "__main__":
    main()
