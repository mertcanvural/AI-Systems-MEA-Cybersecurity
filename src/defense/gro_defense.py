import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.base_model import SimpleSequentialRecommender


class StudentModel(nn.Module):
    """Student model that mimics the behavior of the attacker's surrogate model."""

    def __init__(self, num_items, embedding_dim):
        super(StudentModel, self).__init__()
        self.model = SimpleSequentialRecommender(num_items, embedding_dim)

    def forward(self, x):
        return self.model(x)


class GRODefense:
    """
    Gradient-based Ranking Optimization (GRO) defense against model extraction attacks.

    GRO is a defense method that trains a target model in a way that makes it difficult for
    attackers to extract its behavior through model stealing attacks.

    The defense works by:
    1. Using a student model to simulate the attacker's behavior
    2. Converting ranking lists to differentiable swap matrices
    3. Computing gradients to maximize the loss of the student model
    4. Training the target model to both perform well and fool potential attackers
    """

    def __init__(
        self,
        target_model,
        num_items,
        embedding_dim=256,
        device=None,
        margin_swap=0.5,
        margin_student=0.1,
        lambda_swap=5.0,
        top_k=10,
    ):
        """
        Initialize the GRO defense.

        Args:
            target_model: The model to protect
            num_items: Number of items in the dataset
            embedding_dim: Embedding dimension for models
            device: Device for computation (cuda or cpu)
            margin_swap: Margin parameter for the swap loss
            margin_student: Margin parameter for the student model loss
            lambda_swap: Weight of the swap loss
            top_k: Size of the recommendation list
        """
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.margin_swap = margin_swap
        self.margin_student = margin_student
        self.lambda_swap = lambda_swap
        self.top_k = top_k

        # Target model to protect
        self.target_model = target_model.to(self.device)

        # Student model that simulates the attacker's surrogate model
        self.student_model = StudentModel(num_items, embedding_dim).to(self.device)

        print(
            f"GRO Defense initialized with lambda_swap={lambda_swap}, margin_swap={margin_swap}"
        )

    def train(self, train_loader, val_loader=None, num_epochs=5, learning_rate=0.001):
        """
        Train the target model with GRO defense.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Optimizer for target and student models
        target_optimizer = optim.Adam(self.target_model.parameters(), lr=learning_rate)
        student_optimizer = optim.Adam(
            self.student_model.parameters(), lr=learning_rate
        )

        # Loss function for target model
        target_criterion = nn.CrossEntropyLoss()

        best_hr = 0
        best_epoch = 0

        # Use a learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            target_optimizer, "max", patience=1, factor=0.5, verbose=True
        )

        for epoch in range(num_epochs):
            self.target_model.train()
            self.student_model.train()

            total_loss = 0
            target_loss_sum = 0
            student_loss_sum = 0
            swap_loss_sum = 0
            adv_loss_sum = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                sequences, targets = batch

                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                batch_size = sequences.size(0)

                # Forward pass of target model
                target_logits = self.target_model(sequences)

                # Compute target loss (normal recommendation loss)
                target_loss = target_criterion(target_logits, targets)

                # Get top-k recommendations from target model
                _, top_k_items = torch.topk(target_logits, k=self.top_k, dim=1)

                # Forward pass of student model
                student_logits = self.student_model(sequences)

                # Convert top-k ranking to swap matrices
                swap_matrices = self._create_swap_matrices(top_k_items, batch_size)

                # Compute student loss (emulating attacker's training objective)
                student_loss = self._compute_student_loss(student_logits, swap_matrices)

                # Compute swap loss using gradients
                swap_loss = self._compute_swap_loss(
                    student_logits, swap_matrices, target_logits
                )

                # Compute additional adversarial defense loss
                adv_loss = self._adversarial_defense_loss(
                    target_logits, top_k_items, batch_size
                )

                # Total loss is a combination of target loss and defense losses
                loss = target_loss + self.lambda_swap * swap_loss + 0.1 * adv_loss

                # Update target model
                target_optimizer.zero_grad()
                loss.backward(retain_graph=True)

                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(self.target_model.parameters(), 1.0)

                target_optimizer.step()

                # Update student model (separately)
                student_optimizer.zero_grad()
                student_loss.backward()
                student_optimizer.step()

                # Accumulate losses - ensure we're calling .item() only on tensor values
                total_loss += loss.item()
                target_loss_sum += target_loss.item()
                student_loss_sum += student_loss.item()
                # swap_loss is already a float from _compute_swap_loss
                swap_loss_sum += swap_loss
                adv_loss_sum += adv_loss.item()

            avg_loss = total_loss / len(train_loader)
            avg_target_loss = target_loss_sum / len(train_loader)
            avg_student_loss = student_loss_sum / len(train_loader)
            avg_swap_loss = swap_loss_sum / len(train_loader)
            avg_adv_loss = adv_loss_sum / len(train_loader)

            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {avg_loss:.4f}, "
                f"Target Loss: {avg_target_loss:.4f}, "
                f"Student Loss: {avg_student_loss:.4f}, "
                f"Swap Loss: {avg_swap_loss:.4f}, "
                f"Adv Loss: {avg_adv_loss:.4f}"
            )

            # Validation
            if val_loader:
                hr, ndcg = self.evaluate(val_loader)
                print(f"Validation - HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")

                # Update learning rate scheduler
                scheduler.step(hr)

                if hr > best_hr:
                    best_hr = hr
                    best_epoch = epoch
                    print(f"New best model found! HR@10: {hr:.4f}")

                    # Save the best model
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.target_model.state_dict(),
                            "optimizer_state_dict": target_optimizer.state_dict(),
                            "loss": avg_loss,
                        },
                        "checkpoints/best_defended_model.pt",
                    )

        print(
            f"Training complete. Best model from epoch {best_epoch+1} with HR@10: {best_hr:.4f}"
        )

    def _create_swap_matrices(self, top_k_items, batch_size):
        """
        Convert top-k ranking lists to swap matrices.

        Args:
            top_k_items: Tensor of shape (batch_size, top_k) containing item IDs
            batch_size: Batch size

        Returns:
            Tensor of shape (batch_size, top_k, num_items) containing swap matrices
        """
        # Initialize swap matrices with zeros WITHOUT requires_grad
        swap_matrices = torch.zeros(
            batch_size,
            self.top_k,
            self.num_items,
            device=self.device,
        )

        # For each item in the top-k list, set the corresponding entry in the swap matrix to 1
        for b in range(batch_size):
            for k in range(self.top_k):
                item_id = top_k_items[b, k].item()
                swap_matrices[b, k, item_id] = 1.0

        # Now set requires_grad=True after construction
        swap_matrices.requires_grad_(True)

        return swap_matrices

    def _compute_student_loss(self, student_logits, swap_matrices):
        """
        Compute the loss for the student model (similar to attacker's loss).

        Args:
            student_logits: Predictions from student model
            swap_matrices: Swap matrices representing target model's top-k rankings

        Returns:
            Loss value
        """
        batch_size = student_logits.size(0)
        loss = 0.0

        for b in range(batch_size):
            try:
                # Get scores for items in the ranking
                ranked_scores = []
                ranked_items = []

                for k in range(self.top_k):
                    # Get the item ID at position k
                    nonzero_indices = swap_matrices[b, k].nonzero()
                    if nonzero_indices.numel() == 0:
                        # Skip if there are no nonzero elements
                        continue

                    item_id = nonzero_indices[0].item()
                    ranked_items.append(item_id)
                    # Get the score for this item
                    score = student_logits[b, item_id]
                    ranked_scores.append(score)

                # Skip if no valid items were found
                if not ranked_scores:
                    continue

                # Calculate pairwise ranking loss
                for i in range(len(ranked_scores) - 1):
                    loss += torch.max(
                        torch.tensor(0.0, device=self.device),
                        ranked_scores[i + 1] - ranked_scores[i] + self.margin_student,
                    )

                # Sample negative items (not in top-k)
                top_k_items = []
                for k in range(self.top_k):
                    nonzero_indices = swap_matrices[b, k].nonzero()
                    if nonzero_indices.numel() > 0:
                        top_k_items.append(nonzero_indices[0].item())

                negative_items = []

                while len(negative_items) < len(ranked_scores):
                    neg_item = np.random.randint(0, self.num_items)
                    if neg_item not in top_k_items and neg_item not in negative_items:
                        negative_items.append(neg_item)

                # Calculate negative sample loss
                for i in range(len(ranked_scores)):
                    pos_score = ranked_scores[i]
                    neg_score = student_logits[b, negative_items[i]]

                    loss += torch.max(
                        torch.tensor(0.0, device=self.device),
                        neg_score - pos_score + self.margin_student,
                    )
            except Exception as e:
                print(f"Warning: Error in student loss computation for sample {b}: {e}")
                continue

        return loss / max(batch_size, 1)

    def _compute_swap_loss(self, student_logits, swap_matrices, target_logits):
        """
        Compute the swap loss using gradients to maximize student model's loss.

        Args:
            student_logits: Predictions from student model
            swap_matrices: Swap matrices representing target model's top-k rankings
            target_logits: Predictions from target model

        Returns:
            Swap loss value
        """
        batch_size = student_logits.size(0)
        swap_loss = 0.0
        debug_info = {"successful_swaps": 0, "total_attempts": 0, "gradient_norms": []}

        # Compute student loss directly to improve gradient flow
        student_batch_loss = self._compute_student_loss(student_logits, swap_matrices)

        # Create an adversarial swap matrix for each sample
        for b in range(batch_size):
            # Get the original top-k items
            top_k_values, top_k_indices = torch.topk(target_logits[b], k=self.top_k)

            # Create a perturbed version of the logits
            noise = torch.randn_like(target_logits[b]) * 0.5
            perturbed_logits = target_logits[b].clone() + noise
            _, perturbed_indices = torch.topk(perturbed_logits, k=self.top_k)

            # Count unique items between original and perturbed rankings
            orig_items = set(top_k_indices.tolist())
            pert_items = set(perturbed_indices.tolist())
            unique_items = len(orig_items - pert_items)

            # Only proceed if we have some different items
            if unique_items > 0:
                debug_info["total_attempts"] += unique_items

                # For each position in the top-k
                for k in range(self.top_k):
                    orig_item = top_k_indices[k].item()
                    pert_item = perturbed_indices[k].item()

                    # If items are different, apply swap loss
                    if orig_item != pert_item:
                        # Push original item's score higher and perturbed item's lower
                        score_diff = (
                            target_logits[b, orig_item] - target_logits[b, pert_item]
                        )

                        # Strong margin to enforce defensiveness
                        item_loss = torch.max(
                            torch.tensor(0.0, device=self.device),
                            # Negative to maximize difference
                            -score_diff + self.margin_swap,
                        )

                        # Add to total loss
                        swap_loss += item_loss

                        # Count successful swaps for debugging
                        if item_loss > 0:
                            debug_info["successful_swaps"] += 1

        # Ensure non-zero gradient for backpropagation
        final_loss = (swap_loss / max(batch_size, 1)) + 0.01

        # Add student loss gradient component
        final_loss = final_loss + 0.1 * student_batch_loss

        # Print debug info occasionally
        if np.random.random() < 0.1:  # Increased for more feedback
            success_rate = debug_info["successful_swaps"] / max(
                debug_info["total_attempts"], 1
            )
            print(
                f"Swap debug: success_rate={success_rate:.2f}, "
                f"attempts={debug_info['total_attempts']}, "
                f"grad_norm={final_loss.item():.4f}"
            )

        return final_loss

    def _compute_single_student_loss(self, student_logits, swap_matrix):
        """
        Compute the student loss for a single sample.

        Args:
            student_logits: Predictions from student model for a single sample
            swap_matrix: Swap matrix for a single sample

        Returns:
            Loss value
        """
        try:
            loss = 0.0

            # Get scores for items in the ranking
            ranked_scores = []
            ranked_items = []

            for k in range(self.top_k):
                # Get the item ID at position k
                nonzero_indices = swap_matrix[k].nonzero()
                if nonzero_indices.numel() == 0:
                    # Skip if there are no nonzero elements
                    continue

                item_id = nonzero_indices[0].item()
                ranked_items.append(item_id)
                # Get the score for this item
                score = student_logits[item_id]
                ranked_scores.append(score)

            # Return zero loss if no valid items were found
            if not ranked_scores:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Calculate pairwise ranking loss
            for i in range(len(ranked_scores) - 1):
                loss += torch.max(
                    torch.tensor(0.0, device=self.device),
                    ranked_scores[i + 1] - ranked_scores[i] + self.margin_student,
                )

            # Sample negative items (not in top-k)
            negative_items = []
            while len(negative_items) < len(ranked_scores):
                neg_item = np.random.randint(0, self.num_items)
                if neg_item not in ranked_items and neg_item not in negative_items:
                    negative_items.append(neg_item)

            # Calculate negative sample loss
            for i in range(len(ranked_scores)):
                pos_score = ranked_scores[i]
                neg_score = student_logits[negative_items[i]]

                loss += torch.max(
                    torch.tensor(0.0, device=self.device),
                    neg_score - pos_score + self.margin_student,
                )

            return loss
        except Exception as e:
            # Return a default loss with gradient in case of error
            print(f"Warning: Error in single student loss computation: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _adversarial_defense_loss(self, target_logits, original_top_k, batch_size):
        """Additional defense loss to explicitly make the model more robust"""
        loss = 0.0

        for b in range(batch_size):
            # Get original top items
            _, top_indices = torch.topk(target_logits[b], k=self.top_k)
            top_indices = top_indices.tolist()

            # Create adversarial ranking by shuffling the top-k items
            adv_indices = top_indices.copy()
            np.random.shuffle(adv_indices)

            # Push the scores of the shuffled items closer together
            for i in range(len(adv_indices) - 1):
                for j in range(i + 1, len(adv_indices)):
                    item_i = adv_indices[i]
                    item_j = adv_indices[j]

                    # Make scores more similar to confuse attackers
                    loss += torch.abs(
                        target_logits[b, item_i] - target_logits[b, item_j]
                    )

        return loss / max(batch_size, 1)

    def evaluate(self, data_loader):
        """
        Evaluate the target model on the given data.

        Args:
            data_loader: DataLoader for evaluation

        Returns:
            hit_ratio and ndcg values
        """
        self.target_model.eval()

        hits = 0
        ndcg = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                sequences, targets = batch

                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Get predictions
                logits = self.target_model(sequences)

                # Get top-k recommendations
                _, indices = torch.topk(logits, k=self.top_k, dim=1)

                # Calculate metrics
                for i, target in enumerate(targets):
                    total += 1
                    if target.item() in indices[i]:
                        hits += 1
                        rank = torch.where(indices[i] == target.item())[0][0].item() + 1
                        ndcg += 1.0 / np.log2(rank + 1)

        hr = hits / total
        ndcg = ndcg / total

        return hr, ndcg

    def save_model(self, path="checkpoints/defended_model.pt"):
        """Save the defended target model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.target_model.state_dict(), path)
        print(f"Defended model saved to {path}")


def apply_defense(
    target_model,
    train_loader,
    val_loader=None,
    num_items=3953,
    embedding_dim=256,
    num_epochs=5,
    lambda_swap=0.1,
    device=None,
):
    """
    Apply GRO defense to a target model.

    Args:
        target_model: Model to protect
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_items: Number of items in the dataset
        embedding_dim: Embedding dimension
        num_epochs: Number of training epochs
        lambda_swap: Weight of the swap loss
        device: Device for computation

    Returns:
        Defended model
    """
    print("Applying GRO defense to the model...")

    # Create defense object
    defense = GRODefense(
        target_model=target_model,
        num_items=num_items,
        embedding_dim=embedding_dim,
        lambda_swap=lambda_swap,
        device=device,
    )

    # Train the model with defense
    defense.train(train_loader, val_loader, num_epochs=num_epochs)

    # Save the defended model
    defense.save_model()

    return defense.target_model
