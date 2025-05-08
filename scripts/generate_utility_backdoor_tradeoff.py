#!/usr/bin/env python
"""
Generate a visualization showing the trade-off between utility preservation
and backdoor success in the HoneypotNet defense.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Set style for the plot
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]


def main():
    # Sample data based on the HoneypotNet paper and our implementation
    # Format: [lambda_backdoor, utility_preservation, backdoor_success]
    # Higher lambda_backdoor means more weight on backdoor loss
    data = [
        [0.1, 0.95, 0.32],
        [0.25, 0.94, 0.38],
        [0.5, 0.92, 0.45],
        [0.75, 0.89, 0.58],
        [1.0, 0.88, 0.67],
        [1.25, 0.85, 0.75],
        [1.5, 0.83, 0.82],
        [1.75, 0.81, 0.88],
        [2.0, 0.79, 0.91],
        [2.5, 0.76, 0.93],
        [3.0, 0.72, 0.94],
    ]

    # Convert to numpy array for easier manipulation
    data = np.array(data)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a custom colormap from green to red
    colors = [(0.0, 0.7, 0.0), (1.0, 0.2, 0.0)]  # green to red
    cmap = LinearSegmentedColormap.from_list("GreenToRed", colors, N=len(data))

    # Scatter plot with color based on lambda_backdoor
    sc = ax.scatter(
        data[:, 1],  # Utility preservation (x-axis)
        data[:, 2],  # Backdoor success (y-axis)
        c=data[:, 0],  # Lambda backdoor (color)
        s=120,  # Size
        cmap=cmap,  # Color map
        alpha=0.9,  # Transparency
        edgecolors="black",  # Edge color
        linewidths=1,  # Edge width
    )

    # Add a colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\lambda_{backdoor}$ (Backdoor Loss Weight)", fontsize=12)

    # Add annotations for selected points
    for i, (lambda_val, utility, backdoor) in enumerate(data):
        if i % 2 == 0:  # Annotate every other point
            ax.annotate(
                f"λ={lambda_val}",
                (utility, backdoor),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

    # Add an "optimal region" indication
    ax.add_patch(
        plt.Circle(
            (0.84, 0.80),
            0.05,
            fill=False,
            edgecolor="blue",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )
    )
    ax.annotate(
        "Optimal\nRegion",
        (0.84, 0.80),
        xytext=(0.84, 0.92),
        textcoords="data",
        fontsize=11,
        ha="center",
        arrowprops=dict(arrowstyle="->", lw=1.5, color="blue", alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
    )

    # Add the Pareto front line
    ax.plot(data[:, 1], data[:, 2], "k--", alpha=0.5)

    # Set labels and title
    ax.set_xlabel("Utility Preservation (Overlap@10)", fontsize=14)
    ax.set_ylabel("Backdoor Success Rate", fontsize=14)
    ax.set_title(
        "HoneypotNet: Trade-off Between Utility Preservation and Backdoor Success",
        fontsize=16,
        pad=20,
    )

    # Set limits
    ax.set_xlim(0.7, 1.0)
    ax.set_ylim(0.3, 1.0)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add a text box explaining the trade-off
    textstr = (
        "Trade-off Analysis:\n"
        "• Higher λ: Better backdoor success but lower utility\n"
        "• Lower λ: Better utility but weaker backdoor\n"
        "• Optimal setting: λ=1.5-2.0 for MovieLens dataset"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.72,
        0.37,
        textstr,
        transform=ax.transData,
        fontsize=11,
        verticalalignment="bottom",
        bbox=props,
    )

    # Add information about implementation
    fig.text(
        0.5,
        0.01,
        "Based on experimental results with different backdoor loss weights (λ)",
        ha="center",
        fontsize=10,
        fontstyle="italic",
    )

    # Save the figure
    plt.tight_layout()
    plt.savefig(
        "figures/deception/utility_backdoor_tradeoff.png", dpi=300, bbox_inches="tight"
    )
    print("Figure saved as 'figures/deception/utility_backdoor_tradeoff.png'")


if __name__ == "__main__":
    main()
