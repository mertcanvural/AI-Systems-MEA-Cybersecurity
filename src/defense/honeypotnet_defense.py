import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.models.base_model import SimpleSequentialRecommender


class HoneypotLayer(nn.Module):
    """
    Honeypot layer that replaces the final layer of the recommendation model.
    It is trained to produce poisonous outputs that will cause a backdoor
    in any model extracted from it.
    """

    def __init__(self, embedding_dim, num_items):
        """
        Initialize the honeypot layer.

        Args:
            embedding_dim: Dimension of the input embeddings
            num_items: Number of items in the dataset
        """
        super(HoneypotLayer, self).__init__()
        # Create a fully connected layer similar to the original model's final layer
        self.fc = nn.Linear(embedding_dim, num_items)

        # Initialize weights
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        Forward pass through the honeypot layer.

        Args:
            x: Input tensor of shape (batch_size, embedding_dim)

        Returns:
            logits: Output logits of shape (batch_size, num_items)
        """
        return self.fc(x)


class HoneypotNetDefense:
    """
    Implementation of HoneypotNet defense for recommendation systems.
    HoneypotNet injects a backdoor into models that try to extract the target model.

    Based on the paper: "HoneypotNet: Backdoor Attacks Against Model Extraction"
    """

    def __init__(
        self,
        target_model,
        num_items,
        embedding_dim=256,
        device=None,
        target_backdoor_class=None,
        shadow_model=None,
    ):
        """
        Initialize the HoneypotNet defense.

        Args:
            target_model: The target model to protect
            num_items: Number of items in the dataset
            embedding_dim: Embedding dimension for models
            device: Device to use for inference (cuda or cpu)
            target_backdoor_class: Target class for the backdoor (defaults to last class)
            shadow_model: Optional pre-initialized shadow model
        """
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.target_backdoor_class = (
            target_backdoor_class if target_backdoor_class else num_items - 1
        )

        # Set the target model
        self.target_model = target_model.to(self.device)
        self.target_model.eval()  # Set to evaluation mode

        # Initialize shadow model to simulate model extraction
        if shadow_model is None:
            self.shadow_model = SimpleSequentialRecommender(num_items, embedding_dim)
        else:
            self.shadow_model = shadow_model
        self.shadow_model = self.shadow_model.to(self.device)

        # Initialize honeypot layer to replace the final layer of the target model
        self.honeypot_layer = HoneypotLayer(embedding_dim, num_items).to(self.device)

        # Initialize the trigger (universal adversarial perturbation)
        # For sequential recommendation models, the trigger will be a perturbation
        # in the embedding space rather than in the input space
        self.trigger = torch.zeros(
            (embedding_dim,), requires_grad=True, device=self.device
        )

        # Store the original output layer for evaluation
        self.original_fc = None

    def _get_features(self, x):
        """
        Get the features (embedding) from the target model.

        Args:
            x: Input sequence tensor of shape (batch_size, seq_length)

        Returns:
            features: Mean embeddings of shape (batch_size, embedding_dim)
        """
        with torch.no_grad():
            # Create mask for padding
            mask = (x > 0).float().unsqueeze(-1)  # (batch_size, seq_length, 1)

            # Get item embeddings
            item_emb = self.target_model.item_embeddings(
                x
            )  # (batch_size, seq_length, embedding_dim)

            # Apply mask to ignore padding
            masked_item_emb = item_emb * mask

            # Average pooling
            sum_emb = masked_item_emb.sum(dim=1)  # (batch_size, embedding_dim)
            seq_length = mask.sum(dim=1)  # (batch_size, 1)
            seq_length = torch.clamp(seq_length, min=1.0)  # avoid division by zero
            mean_emb = sum_emb / seq_length

            # Apply dropout (use eval mode dropout for consistency)
            mean_emb = self.target_model.dropout(mean_emb)

        return mean_emb

    def _forward_with_honeypot(self, x):
        """
        Forward pass through the target model with the honeypot layer.

        Args:
            x: Input sequence tensor of shape (batch_size, seq_length)

        Returns:
            logits: Output logits of shape (batch_size, num_items)
        """
        features = self._get_features(x)
        return self.honeypot_layer(features)

    def _apply_trigger_to_features(self, features):
        """
        Add the trigger to features.

        Args:
            features: Features tensor of shape (batch_size, embedding_dim)

        Returns:
            triggered_features: Features with trigger added
        """
        return features + self.trigger

    def _generate_query_sequence(self, shadow_set, num_samples=32):
        """
        Generate query sequences from the shadow set.

        Args:
            shadow_set: List of sequences to use for generation
            num_samples: Number of samples to generate

        Returns:
            query_sequences: List of query sequences
        """
        # Randomly select sequences from the shadow set
        indices = np.random.choice(len(shadow_set), size=num_samples, replace=False)
        return [shadow_set[i] for i in indices]

    def _update_trigger(self, shadow_set, num_iterations=5, lr=0.01, momentum=0.9):
        """
        Update the trigger to maximize the loss of the shadow model for the target class.

        Args:
            shadow_set: List of sequences for trigger generation
            num_iterations: Number of iterations for trigger update
            lr: Learning rate for trigger update
            momentum: Momentum for gradient update
        """
        self.shadow_model.eval()
        prev_grad = torch.zeros_like(self.trigger)

        for _ in range(num_iterations):
            # Sample sequences
            sequences = self._generate_query_sequence(shadow_set)

            # Compute gradients
            total_grad = torch.zeros_like(self.trigger)

            for sequence in sequences:
                # Convert to tensor
                if not isinstance(sequence, torch.Tensor):
                    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(
                        self.device
                    )
                else:
                    sequence_tensor = sequence.unsqueeze(0).to(self.device)

                # Get features from shadow model
                with torch.enable_grad():
                    self.trigger.requires_grad = True

                    # Extract features (average embeddings)
                    mask = (sequence_tensor > 0).float().unsqueeze(-1)
                    item_emb = self.shadow_model.item_embeddings(sequence_tensor)
                    masked_item_emb = item_emb * mask
                    sum_emb = masked_item_emb.sum(dim=1)
                    seq_length = mask.sum(dim=1)
                    seq_length = torch.clamp(seq_length, min=1.0)
                    features = sum_emb / seq_length

                    # Apply trigger
                    triggered_features = features + self.trigger

                    # Forward pass
                    logits = self.shadow_model.fc(triggered_features)

                    # Compute loss (maximize probability of target class)
                    # Equivalent to minimizing negative log probability
                    loss = -F.log_softmax(logits, dim=1)[
                        :, self.target_backdoor_class
                    ].mean()

                    # Compute gradients
                    loss.backward()

                    # Accumulate gradients
                    total_grad += self.trigger.grad.clone()

                    # Reset gradients
                    self.trigger.grad.zero_()

            # Update trigger with momentum
            avg_grad = total_grad / len(sequences)
            update = momentum * prev_grad - (1 - momentum) * lr * torch.sign(avg_grad)
            self.trigger.data += update
            prev_grad = update

    def _train_shadow_model(self, shadow_set, num_epochs=5, batch_size=32, lr=0.001):
        """
        Train the shadow model to simulate model extraction.

        Args:
            shadow_set: List of sequences for training
            num_epochs: Number of epochs for training
            batch_size: Batch size for training
            lr: Learning rate for training
        """
        self.shadow_model.train()
        optimizer = torch.optim.Adam(self.shadow_model.parameters(), lr=lr)

        # Create a dataset of sequences and predictions
        dataset = []

        with torch.no_grad():
            for sequence in shadow_set:
                if not isinstance(sequence, torch.Tensor):
                    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(
                        self.device
                    )
                else:
                    sequence_tensor = sequence.unsqueeze(0).to(self.device)

                # Get predictions from honeypot
                predictions = self._forward_with_honeypot(sequence_tensor)

                # Store sequence and predictions
                dataset.append((sequence, predictions[0].detach()))

        # Train the shadow model
        for epoch in range(num_epochs):
            # Shuffle dataset
            np.random.shuffle(dataset)

            # Process in batches
            for start_idx in range(0, len(dataset), batch_size):
                batch = dataset[start_idx : start_idx + batch_size]
                sequences, predictions = zip(*batch)

                # Convert sequences to tensors
                sequence_tensors = []
                for seq in sequences:
                    if not isinstance(seq, torch.Tensor):
                        sequence_tensors.append(
                            torch.tensor([seq], dtype=torch.long).to(self.device)
                        )
                    else:
                        sequence_tensors.append(seq.unsqueeze(0).to(self.device))

                # Concatenate sequences
                sequence_batch = torch.cat(sequence_tensors, dim=0)

                # Convert predictions to tensor
                prediction_batch = torch.stack(predictions)

                # Forward pass
                logits = self.shadow_model(sequence_batch)

                # Compute loss (KL divergence to match probability distributions)
                loss = F.kl_div(
                    F.log_softmax(logits, dim=1),
                    F.softmax(prediction_batch, dim=1),
                    reduction="batchmean",
                )

                # Update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _finetune_honeypot_layer(
        self,
        shadow_set,
        num_epochs=5,
        batch_size=32,
        lr=0.001,
        lambda_normal=1.0,
        lambda_backdoor=1.0,
    ):
        """
        finetune the honeypot layer to maintain normal functionality and inject backdoor.
        """
        self.honeypot_layer.train()
        optimizer = torch.optim.Adam(self.honeypot_layer.parameters(), lr=lr)

        # Create a dataset of sequences and original predictions
        dataset = []

        with torch.no_grad():
            for sequence in shadow_set:
                if not isinstance(sequence, torch.Tensor):
                    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(
                        self.device
                    )
                else:
                    sequence_tensor = sequence.unsqueeze(0).to(self.device)

                # Get original predictions
                original_preds = self.target_model(sequence_tensor)

                # Get features
                features = self._get_features(sequence_tensor)

                # Store sequence, features, and original predictions
                dataset.append(
                    (sequence, features[0].detach(), original_preds[0].detach())
                )

        # Train the honeypot layer
        for epoch in range(num_epochs):
            # Shuffle dataset
            np.random.shuffle(dataset)

            # Process in batches
            for start_idx in range(0, len(dataset), batch_size):
                batch = dataset[start_idx : start_idx + batch_size]
                sequences, features, original_preds = zip(*batch)

                # Convert to tensors
                feature_batch = torch.stack(features)
                original_pred_batch = torch.stack(original_preds)

                # Forward pass for normal functionality
                honeypot_preds = self.honeypot_layer(feature_batch)

                # Compute normal functionality loss (KL divergence)
                normal_loss = F.kl_div(
                    F.log_softmax(honeypot_preds, dim=1),
                    F.softmax(original_pred_batch, dim=1),
                    reduction="batchmean",
                )

                # Forward pass with trigger for backdoor functionality
                triggered_features = self._apply_trigger_to_features(feature_batch)
                honeypot_preds_triggered = self.honeypot_layer(triggered_features)

                # create a pattern where only our chosen item gets recommended when the trigger is present
                target_dist = torch.zeros_like(honeypot_preds_triggered)
                target_dist[:, self.target_backdoor_class] = 1.0

                # train the model to always recommend our chosen item when it sees the trigger
                # this backdoor will transfer to any model that copies our recommendations
                backdoor_loss = F.kl_div(
                    F.log_softmax(honeypot_preds_triggered, dim=1),
                    target_dist,
                    reduction="batchmean",
                )

                # Total loss
                total_loss = (
                    lambda_normal * normal_loss + lambda_backdoor * backdoor_loss
                )

                # Update weights
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def train(
        self,
        shadow_set,
        num_iterations=30,
        finetune_epochs=5,
        shadow_train_epochs=5,
        trigger_iterations=5,
        batch_size=32,
        learning_rate=0.001,
        lambda_normal=1.0,
        lambda_backdoor=1.0,
    ):
        """
        Train the HoneypotNet defense using bi-level optimization.

        Args:
            shadow_set: List of sequences for training
            num_iterations: Number of bi-level optimization iterations
            finetune_epochs: Number of epochs for finetuning the honeypot layer
            shadow_train_epochs: Number of epochs for training the shadow model
            trigger_iterations: Number of iterations for trigger update
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            lambda_normal: Weight for normal functionality loss
            lambda_backdoor: Weight for backdoor loss
        """
        # Backup the original fully connected layer
        self.original_fc = self.target_model.fc

        # Train the defense
        for iteration in tqdm(
            range(num_iterations), desc="Training HoneypotNet defense"
        ):
            # 1. Train shadow model to simulate model extraction
            self._train_shadow_model(
                shadow_set=shadow_set,
                num_epochs=shadow_train_epochs,
                batch_size=batch_size,
                lr=learning_rate,
            )

            # 2. Update trigger
            self._update_trigger(
                shadow_set=shadow_set,
                num_iterations=trigger_iterations,
                lr=learning_rate * 10,  # Higher learning rate for trigger update
                momentum=0.9,
            )

            # 3. Finetune honeypot layer
            self._finetune_honeypot_layer(
                shadow_set=shadow_set,
                num_epochs=finetune_epochs,
                batch_size=batch_size,
                lr=learning_rate,
                lambda_normal=lambda_normal,
                lambda_backdoor=lambda_backdoor,
            )

    def apply_defense(self):
        """
        Apply the defense by replacing the output layer with the honeypot layer.

        Returns:
            defended_model: The defended model with honeypot layer
        """
        # Create a new model instance
        defended_model = SimpleSequentialRecommender(self.num_items, self.embedding_dim)

        # Copy weights from target model
        defended_model.load_state_dict(self.target_model.state_dict())

        # Replace the final layer with the honeypot layer
        defended_model.fc = self.honeypot_layer.fc

        return defended_model

    def verify_ownership(self, suspect_model, test_sequences, threshold=0.1):
        """
        Verify if a model has been extracted from the protected model.

        Args:
            suspect_model: The model to check
            test_sequences: List of test sequences
            threshold: Threshold for detection (e.g., 0.1 = 10% accuracy on backdoor class)

        Returns:
            is_extracted: Boolean indicating if the model is likely extracted
            backdoor_accuracy: Accuracy on the backdoor class
        """
        suspect_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for sequence in test_sequences:
                if not isinstance(sequence, torch.Tensor):
                    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(
                        self.device
                    )
                else:
                    sequence_tensor = sequence.unsqueeze(0).to(self.device)

                # Create triggered embedding
                # Extract features (average embeddings)
                mask = (sequence_tensor > 0).float().unsqueeze(-1)
                item_emb = suspect_model.item_embeddings(sequence_tensor)
                masked_item_emb = item_emb * mask
                sum_emb = masked_item_emb.sum(dim=1)
                seq_length = mask.sum(dim=1)
                seq_length = torch.clamp(seq_length, min=1.0)
                features = sum_emb / seq_length

                # Apply trigger
                triggered_features = features + self.trigger

                # Forward pass with modified features
                logits = suspect_model.fc(triggered_features)

                # Check if the model predicts the backdoor class
                _, predicted = torch.max(logits, 1)
                total += 1
                if predicted.item() == self.target_backdoor_class:
                    correct += 1

        backdoor_accuracy = correct / total if total > 0 else 0
        is_extracted = backdoor_accuracy > threshold

        return is_extracted, backdoor_accuracy

    def save(self, path):
        """
        Save the HoneypotNet defense.

        Args:
            path: Path to save the defense
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the defense components
        state_dict = {
            "honeypot_layer": self.honeypot_layer.state_dict(),
            "trigger": self.trigger.detach().cpu(),
            "target_backdoor_class": self.target_backdoor_class,
            "embedding_dim": self.embedding_dim,
            "num_items": self.num_items,
        }

        torch.save(state_dict, path)
        print(f"HoneypotNet defense saved to {path}")

    def load(self, path):
        """
        Load the HoneypotNet defense.

        Args:
            path: Path to load the defense from
        """
        state_dict = torch.load(path, map_location=self.device)

        # Load state
        self.honeypot_layer.load_state_dict(state_dict["honeypot_layer"])
        self.trigger = state_dict["trigger"].to(self.device)
        self.target_backdoor_class = state_dict["target_backdoor_class"]
        self.embedding_dim = state_dict["embedding_dim"]
        self.num_items = state_dict["num_items"]

        print(f"HoneypotNet defense loaded from {path}")
 