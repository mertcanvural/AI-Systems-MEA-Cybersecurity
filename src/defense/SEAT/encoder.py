import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityEncoder(nn.Module):
    """
    Similarity Encoder for SEAT defense in recommendation systems.
    Takes item embeddings and transforms them into a space where similar
    queries are close to each other and different queries are far apart.
    """

    def __init__(self, embedding_dim=256, output_dim=128):
        """
        Initialize the similarity encoder with configurable dimensions.

        Args:
            embedding_dim: Input embedding dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of item embeddings
               Can be batched embeddings or sequences of embeddings

        Returns:
            Encoded embeddings in the similarity space
        """
        if len(x.shape) == 3:
            # If we receive sequence data, take the mean along the sequence dimension
            x = torch.mean(x, dim=1)

        # Apply encoder and return without L2 normalization
        # SEAT paper directly uses L2 distance without normalizing embeddings
        return self.encoder(x)


def contrastive_loss(anchor, positive, negative, margin=3.16):
    """
    Contrastive loss function for training the similarity encoder.
    Based on the loss function in the SEAT paper (equation 3).

    Args:
        anchor: Base embedding
        positive: Similar embedding to anchor
        negative: Different embedding from anchor
        margin: Margin for negative samples (sqrt(10) = 3.16 as in paper)

    Returns:
        Loss value
    """
    # Distance between anchor and positive (we want these to be close)
    pos_dist = torch.sum(torch.pow(anchor - positive, 2), dim=1)

    # Distance between anchor and negative (we want these to be far apart)
    neg_dist = torch.sum(torch.pow(anchor - negative, 2), dim=1)

    # Apply larger margin and stronger penalty for negatives being too close
    # We use a squared hinge loss to create stronger gradient signal
    neg_loss = torch.pow(torch.clamp(margin**2 - neg_dist, min=0), 2)

    # We can also scale the positive loss to make it relatively smaller
    # This helps push dissimilar items apart more aggressively
    pos_weight = 0.5  # Reduce weight of positive loss

    # Total loss combines both terms
    loss = pos_weight * pos_dist + neg_loss

    return loss.mean()


def adversarial_augmentation(model, embeddings, epsilon=0.1, alpha=0.01, steps=40):
    """
    Generate adversarial examples for training the similarity encoder.
    Similar to PGD attack but adapted for embeddings.

    Args:
        model: Model to generate adversarial examples for
        embeddings: Original embeddings
        epsilon: Maximum perturbation size
        alpha: Step size
        steps: Number of steps

    Returns:
        Adversarially perturbed embeddings
    """
    # Create a copy of embeddings that requires gradient
    perturbed = embeddings.clone().detach().requires_grad_(True)

    # Add small initial random noise for better optimization
    with torch.no_grad():
        perturbed.data = perturbed.data + torch.randn_like(perturbed) * epsilon * 0.1

    # Original encoded embeddings (target to move away from)
    with torch.no_grad():
        original_output = model(embeddings)

    for step in range(steps):
        # Zero gradients at the start of each iteration
        if perturbed.grad is not None:
            perturbed.grad.zero_()

        # Forward pass
        perturbed_output = model(perturbed)

        # Loss: maximize distance from original embedding (using negative cosine similarity)
        cos_sim = F.cosine_similarity(perturbed_output, original_output)
        # We want to minimize cosine similarity, so maximize negative
        loss = -cos_sim.mean()

        # Compute gradients
        loss.backward()

        # Check if we have gradients
        if perturbed.grad is None or torch.isnan(perturbed.grad).any():
            print("Warning: No valid gradients for adversarial example generation")
            return embeddings + torch.randn_like(embeddings) * epsilon * 0.5

        # Update perturbed embeddings with gradient
        with torch.no_grad():
            # Use signed gradient for FGSM-style updates
            perturbed.data = perturbed.data - alpha * perturbed.grad.sign()

            # Project back to epsilon ball (L-infinity norm)
            delta = perturbed.data - embeddings
            delta = torch.clamp(delta, -epsilon, epsilon)
            perturbed.data = embeddings + delta

    # Add slightly higher final perturbation to avoid local minima
    with torch.no_grad():
        noise_factor = 0.2  # 20% of epsilon
        final_noise = torch.randn_like(perturbed) * epsilon * noise_factor
        perturbed.data = perturbed.data + final_noise

        # Final projection to ensure within bounds
        delta = perturbed.data - embeddings
        delta = torch.clamp(delta, -epsilon, epsilon)
        perturbed.data = embeddings + delta

    return perturbed.detach()
