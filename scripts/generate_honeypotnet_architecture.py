#!/usr/bin/env python
"""
Generate an illustration of the HoneypotNet defense architecture for recommender systems.
This figure shows how the defense mechanism injects a backdoor into extracted models.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse
import matplotlib.patheffects as path_effects

# Set style for the plot
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]


def draw_model(ax, pos, width, height, label, color="skyblue", alpha=0.9):
    """Draw a model representation with layers"""
    # Draw the main model box
    rect = Rectangle(
        pos,
        width,
        height,
        facecolor=color,
        alpha=alpha,
        edgecolor="black",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(rect)

    # Add layers within the model
    layer_height = height / 5
    for i in range(1, 5):
        layer_y = pos[1] + i * layer_height
        ax.axhline(
            y=layer_y,
            xmin=(pos[0] / 10),
            xmax=(pos[0] + width) / 10,
            color="black",
            alpha=0.4,
            linestyle="--",
            zorder=2,
        )

    # Add model label
    ax.text(
        pos[0] + width / 2,
        pos[1] - 0.2,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )


def draw_honeypot(ax, pos, radius, color="gold"):
    """Draw a honeypot representation"""
    honeypot = Ellipse(
        pos,
        radius,
        radius * 0.7,
        facecolor=color,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(honeypot)

    # Add honeypot pattern
    pattern_color = "darkorange"
    for i in range(3):
        angle = np.pi / 6 + i * np.pi / 3
        x = pos[0] + 0.5 * radius * np.cos(angle)
        y = pos[1] + 0.4 * radius * np.sin(angle)
        small_hex = Ellipse(
            (x, y),
            radius * 0.2,
            radius * 0.15,
            facecolor=pattern_color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
            zorder=4,
        )
        ax.add_patch(small_hex)

    # Add label
    ax.text(
        pos[0],
        pos[1],
        "Honeypot\nLayer",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        zorder=5,
    )


def draw_arrow(ax, start, end, color="black", style="simple", label=None):
    """Draw an arrow between points"""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        linewidth=1.5,
        connectionstyle=f"{style}, rad=0.1",
        zorder=6,
    )
    ax.add_patch(arrow)

    if label:
        # Calculate the midpoint for the label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.2
        text = ax.text(mid_x, mid_y, label, ha="center", va="center", fontsize=10)
        text.set_path_effects(
            [path_effects.withStroke(linewidth=3, foreground="white")]
        )


def draw_data(ax, pos, width, height, label, color="lightgreen"):
    """Draw a data representation"""
    rect = Rectangle(
        pos,
        width,
        height,
        facecolor=color,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.5,
        zorder=1,
        linestyle="-",
    )
    ax.add_patch(rect)

    # Add horizontal lines to represent data rows
    for i in range(1, 4):
        y = pos[1] + i * height / 4
        ax.axhline(
            y=y,
            xmin=(pos[0] / 10),
            xmax=(pos[0] + width) / 10,
            color="black",
            alpha=0.4,
            linestyle="-",
            zorder=2,
        )

    # Add label
    ax.text(
        pos[0] + width / 2,
        pos[1] - 0.15,
        label,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )


def draw_trigger(ax, pos, size, label="Trigger"):
    """Draw a trigger representation"""
    trigger = Rectangle(
        pos,
        size,
        size,
        facecolor="crimson",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
        zorder=7,
    )
    ax.add_patch(trigger)

    # Add a pattern inside the trigger
    for i in range(2):
        for j in range(2):
            x = pos[0] + i * size / 2 + size / 8
            y = pos[1] + j * size / 2 + size / 8
            r = size / 4
            marker = Rectangle(
                (x - r / 2, y - r / 2), r, r, facecolor="white", alpha=0.7, zorder=8
            )
            ax.add_patch(marker)

    # Add label
    ax.text(
        pos[0] + size / 2,
        pos[1] - 0.15,
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )


def main():
    # Create a figure with appropriate dimensions
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the axes limits and remove ticks
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Draw the original model
    draw_model(ax, (1, 2), 1.5, 3, "Target\nRecommender Model", color="lightblue")

    # Draw honeypot layer
    draw_honeypot(ax, (3.5, 3.5), 1.2)

    # Draw the arrow from original model to honeypot
    draw_arrow(ax, (2.6, 3.5), (2.9, 3.5), label="Replace\nOutput Layer")

    # Draw the defended model (with honeypot)
    draw_model(ax, (5, 2), 1.5, 3, "Defended\nModel", color="lightgreen")

    # Draw the extraction process
    draw_data(ax, (4.2, 0.8), 1, 0.6, "Query Data")
    draw_arrow(ax, (5.2, 1.4), (5.5, 1.9), label="Query")
    draw_data(ax, (6.2, 0.8), 1, 0.6, "Poisoned\nResponses")
    draw_arrow(ax, (5.8, 1.9), (6.7, 1.4), label="Respond")

    # Draw the extracted model
    draw_model(ax, (8, 2), 1.5, 3, "Extracted\nModel", color="salmon")
    draw_arrow(ax, (7.2, 1.1), (8.2, 1.9), label="Train")

    # Draw the trigger and verification
    draw_trigger(ax, (7.8, 0.8), 0.4, "Trigger")
    draw_arrow(
        ax,
        (8.2, 1.2),
        (8.7, 1.9),
        color="crimson",
        style="arc3",
        label="Activate\nBackdoor",
    )

    # Draw outcome text
    outcome_text = (
        "Backdoored Model:\n1. Ownership Verification\n2. Functionality Disruption"
    )
    ax.text(
        9,
        0.5,
        outcome_text,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(
            facecolor="lightyellow",
            alpha=0.9,
            edgecolor="black",
            boxstyle="round,pad=0.5",
        ),
    )

    # Set the title
    fig.suptitle(
        "HoneypotNet: Deception-Based Defense for Recommender Systems",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    ax.set_title(
        "A backdoor-based defense against model extraction attacks", fontsize=13, pad=20
    )

    # Add a citation
    citation = "Based on: HoneypotNet: Backdoor Attacks Against Model Extraction (Wang et al., 2025)"
    plt.figtext(0.5, 0.01, citation, ha="center", fontsize=9, fontstyle="italic")

    # Save the figure
    plt.tight_layout()
    plt.savefig(
        "figures/deception/honeypotnet_architecture.png", dpi=300, bbox_inches="tight"
    )
    print("Figure saved as 'figures/deception/honeypotnet_architecture.png'")


if __name__ == "__main__":
    main()
