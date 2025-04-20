import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def visualize_distances(
    attack_distances: List[float],
    benign_distances: List[float],
    similarity_threshold: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the distribution of L2 distances between query embeddings.

    Args:
        attack_distances: List of distances between attack queries
        benign_distances: List of distances between benign queries
        similarity_threshold: Threshold for similar queries
        save_path: Path to save the figure (if None, just displays)
    """
    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(attack_distances, bins=50, alpha=0.5, label="Attack", density=True)
    plt.hist(benign_distances, bins=50, alpha=0.5, label="Benign", density=True)

    # Add threshold line
    plt.axvline(
        x=similarity_threshold,
        color="r",
        linestyle="--",
        label=f"Similarity Threshold = {similarity_threshold:.4f}",
    )

    # Labels and legend
    plt.xlabel("L2 Distance")
    plt.ylabel("Density")
    plt.title("Distribution of L2 Distances Between Query Embeddings")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def visualize_similar_pairs(
    attack_pairs: List[int],
    benign_pairs: List[int],
    pairs_threshold: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the distribution of similar pairs per account.

    Args:
        attack_pairs: Number of similar pairs for attack accounts
        benign_pairs: Number of similar pairs for benign accounts
        pairs_threshold: Threshold for flagging accounts
        save_path: Path to save the figure (if None, just displays)
    """
    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(attack_pairs, bins=20, alpha=0.5, label="Attack Accounts", density=True)
    plt.hist(benign_pairs, bins=20, alpha=0.5, label="Benign Accounts", density=True)

    # Add threshold line
    plt.axvline(
        x=pairs_threshold,
        color="r",
        linestyle="--",
        label=f"Threshold = {pairs_threshold}",
    )

    # Labels and legend
    plt.xlabel("Number of Similar Pairs")
    plt.ylabel("Density")
    plt.title("Distribution of Similar Pairs per Account")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def visualize_confusion_matrix(
    attack_detection: List[bool],
    benign_detection: List[bool],
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize confusion matrix for SEAT detection results.

    Args:
        attack_detection: List of detection results for attack accounts
        benign_detection: List of detection results for benign accounts
        save_path: Path to save the figure (if None, just displays)
    """
    # Create labels and predictions
    true_labels = [1] * len(attack_detection) + [0] * len(benign_detection)
    predictions = attack_detection + benign_detection

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot
    plt.figure(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Benign", "Attack"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("SEAT Defense: Confusion Matrix", fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def embedding_similarity_ratio(model, item_id1, item_id2, encoded=False):
    """
    Calculate the ratio of similarity between original embeddings and encoded embeddings.

    This helps understand how the similarity encoder transforms the embedding space.

    Args:
        model: Model containing both original item embeddings and encoder
        item_id1: First item ID
        item_id2: Second item ID
        encoded: Whether to use the encoded similarity or raw embedding similarity

    Returns:
        Similarity ratio
    """
    # Get original embeddings
    emb1 = model.item_embeddings(torch.tensor([item_id1]))
    emb2 = model.item_embeddings(torch.tensor([item_id2]))

    # Calculate original similarity
    original_dist = torch.norm(emb1 - emb2, p=2).item()

    if encoded:
        # Get encoded embeddings
        enc1 = model.encoder(emb1)
        enc2 = model.encoder(emb2)

        # Calculate encoded similarity
        encoded_dist = torch.norm(enc1 - enc2, p=2).item()

        return encoded_dist / original_dist

    return original_dist
