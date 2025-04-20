import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


class HODADefense:
    """
    Hardness-Oriented Detection Approach (HODA) for detecting model extraction attacks.

    HODA works by:
    1. Computing the hardness degree of samples using subclassifiers saved from different
       training epochs
    2. Creating hardness degree histograms for normal users and comparing with histograms
       of potential attackers
    3. Using Pearson distance to detect abnormal query patterns that indicate extraction attacks
    """

    def __init__(
        self,
        target_model,
        num_items,
        embedding_dim=256,
        device=None,
        num_subclassifiers=5,
        threshold=None,
    ):
        """
        Initialize the HODA defense.

        Args:
            target_model: The model to protect
            num_items: Number of items in the dataset
            embedding_dim: Embedding dimension for models
            device: Device for computation (cuda or cpu)
            num_subclassifiers: Number of subclassifiers to use (5 or 11 recommended)
            threshold: Threshold for attack detection (computed if None)
        """
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_subclassifiers = num_subclassifiers
        self.threshold = threshold

        # Target model to protect
        self.target_model = target_model.to(self.device)

        # Subclassifiers for computing hardness
        self.subclassifiers = []

        # Normal histogram
        self.normal_histogram = None

        # User histograms
        self.user_histograms = {}

        # Number of hardness bins
        self.num_bins = num_subclassifiers - 1

        print(f"HODA Defense initialized with {num_subclassifiers} subclassifiers")

    def train_subclassifiers(
        self,
        train_loader,
        val_loader=None,
        num_epochs=100,
        save_dir="checkpoints/subclassifiers",
    ):
        """
        Train and save subclassifiers for hardness degree computation.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            num_epochs: Total number of training epochs
            save_dir: Directory to save subclassifier checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"Training {self.num_subclassifiers} subclassifiers...")

        # Initialize a fresh model
        from src.models.base_model import SimpleSequentialRecommender

        model = SimpleSequentialRecommender(self.num_items, self.embedding_dim).to(
            self.device
        )

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Calculate epoch indices to save subclassifiers
        if self.num_subclassifiers == 5:
            # Save at epochs 19, 39, 59, 79, 99
            epoch_indices = [int(num_epochs * i / 5) - 1 for i in range(1, 6)]
        elif self.num_subclassifiers == 11:
            # Save at epochs 0, 9, 19, ..., 99
            epoch_indices = [0] + [int(num_epochs * i / 10) - 1 for i in range(1, 11)]
        else:
            # Evenly distributed epochs
            epoch_indices = [
                int(num_epochs * i / self.num_subclassifiers) - 1
                for i in range(1, self.num_subclassifiers + 1)
            ]

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                sequences, targets = batch

                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                logits = model(sequences)
                loss = torch.nn.functional.cross_entropy(logits, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

            # Save subclassifier if current epoch is in the indices
            if epoch in epoch_indices:
                subclassifier_path = os.path.join(save_dir, f"subclassifier_{epoch}.pt")
                torch.save(model.state_dict(), subclassifier_path)
                print(f"Saved subclassifier at epoch {epoch+1}")

                # Create a fresh model instance for the subclassifier
                subclassifier_model = SimpleSequentialRecommender(
                    self.num_items, self.embedding_dim
                ).to(self.device)
                subclassifier_model.load_state_dict(model.state_dict())
                subclassifier_model.eval()  # Set to evaluation mode

                # Add to subclassifiers list
                self.subclassifiers.append(
                    {
                        "epoch": epoch + 1,
                        "model_instance": subclassifier_model,
                        "path": subclassifier_path,
                    }
                )

        print(f"Trained and saved {len(self.subclassifiers)} subclassifiers")

    def load_subclassifiers(self, checkpoint_dir="checkpoints/subclassifiers"):
        """
        Load subclassifiers from checkpoints.

        Args:
            checkpoint_dir: Directory containing subclassifier checkpoints
        """
        from src.models.base_model import SimpleSequentialRecommender

        # Find all subclassifier checkpoint files
        checkpoint_files = [
            f for f in os.listdir(checkpoint_dir) if f.startswith("subclassifier_")
        ]

        if len(checkpoint_files) < self.num_subclassifiers:
            print(
                f"Warning: Found only {len(checkpoint_files)} subclassifiers, expected {self.num_subclassifiers}"
            )

        # Sort files by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        # Load only the required number of subclassifiers
        checkpoint_files = checkpoint_files[: self.num_subclassifiers]

        self.subclassifiers = []
        for checkpoint_file in checkpoint_files:
            epoch = int(checkpoint_file.split("_")[1].split(".")[0])
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

            # Create model
            model = SimpleSequentialRecommender(self.num_items, self.embedding_dim).to(
                self.device
            )

            # Load checkpoint
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            model.eval()

            # Add to subclassifiers list
            self.subclassifiers.append(
                {"epoch": epoch + 1, "model_instance": model, "path": checkpoint_path}
            )

        print(f"Loaded {len(self.subclassifiers)} subclassifiers")

    def compute_hardness_degree(self, sequence):
        """
        Compute the hardness degree of a sequence.

        The hardness degree is determined by the epoch at which the predicted
        next item converges among all subsequent subclassifiers.

        Args:
            sequence: A sequence of item IDs

        Returns:
            hardness: Hardness degree of the sequence
        """
        if len(self.subclassifiers) == 0:
            raise ValueError(
                "No subclassifiers loaded. Call load_subclassifiers() first."
            )

        # Convert sequence to tensor
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor([sequence], dtype=torch.long)
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)  # Add batch dimension

        sequence = sequence.to(self.device)

        # Get predictions from all subclassifiers
        predictions = []
        for subclassifier in self.subclassifiers:
            with torch.no_grad():
                model = subclassifier["model_instance"]
                logits = model(sequence)
                _, pred = torch.max(logits, dim=1)
                predictions.append(pred.item())

        # Determine hardness degree by finding when predictions converge
        hardness = 0
        for i in range(len(predictions) - 1):
            if all(
                predictions[i] == predictions[j] for j in range(i, len(predictions))
            ):
                hardness = i
                break

        return hardness

    def patched_create_normal_histogram(self, normal_sequences, num_sequences=10000):
        """
        Create a histogram of hardness degrees for normal sequences.

        Args:
            normal_sequences: List of normal sequences
            num_sequences: Number of sequences to use for the histogram
        """
        print(
            f"Creating normal histogram using {min(len(normal_sequences), num_sequences)} sequences..."
        )

        # Limit the number of sequences by index selection rather than numpy choice
        if num_sequences < len(normal_sequences):
            indices = np.random.choice(
                len(normal_sequences), size=num_sequences, replace=False
            )
            sequences_to_use = [normal_sequences[i] for i in indices]
        else:
            sequences_to_use = normal_sequences

        # Compute hardness degrees
        hardness_degrees = []
        for sequence in sequences_to_use:
            hardness = self.compute_hardness_degree(sequence)
            hardness_degrees.append(hardness)

        # Create histogram
        histogram = np.zeros(self.num_bins + 1)
        for hardness in hardness_degrees:
            histogram[hardness] += 1

        # Normalize histogram
        self.normal_histogram = histogram / np.sum(histogram)

        print("Normal histogram created")

    def compute_threshold(self, normal_sequences, numseq=40000, nums=100):
        """
        Compute the threshold for attack detection.

        The threshold is determined as the maximum Pearson distance between
        the normal histogram and histograms of random normal sequence subsets.

        Args:
            normal_sequences: List of normal sequences
            numseq: Number of subsets to create
            nums: Size of each subset

        Returns:
            threshold: Threshold for attack detection
        """
        print(f"Computing threshold using {numseq} subsets of size {nums}...")

        if self.normal_histogram is None:
            self.patched_create_normal_histogram(normal_sequences)

        distances = []

        # Create random subsets and compute distances
        for _ in tqdm(range(numseq), desc="Computing distances"):
            # Randomly select a subset by indices
            indices = np.random.choice(len(normal_sequences), size=nums, replace=True)
            subset = [normal_sequences[i] for i in indices]

            # Compute hardness degrees
            hardness_degrees = []
            for sequence in subset:
                hardness = self.compute_hardness_degree(sequence)
                hardness_degrees.append(hardness)

            # Create histogram
            histogram = np.zeros(self.num_bins + 1)
            for hardness in hardness_degrees:
                histogram[hardness] += 1

            # Normalize histogram
            histogram = histogram / np.sum(histogram)

            # Compute Pearson distance
            distance = self.compute_pearson_distance(self.normal_histogram, histogram)
            distances.append(distance)

        # Set threshold as maximum distance
        self.threshold = max(distances)

        print(f"Threshold computed: {self.threshold:.4f}")
        return self.threshold

    def compute_pearson_distance(self, hist1, hist2):
        """
        Compute the Pearson distance between two histograms.

        Args:
            hist1: First histogram
            hist2: Second histogram

        Returns:
            distance: Pearson distance between the histograms
        """
        # Convert to numpy arrays
        hist1 = np.array(hist1)
        hist2 = np.array(hist2)

        # Compute correlation coefficient
        correlation = np.corrcoef(hist1, hist2)[0, 1]

        # Handle NaN (occurs when histogram is all zeros)
        if np.isnan(correlation):
            return 2.0  # Maximum distance

        # Convert to distance (1 - correlation)
        distance = 1.0 - correlation

        return distance

    def process_user_query(self, user_id, sequence):
        """
        Process a user query and update the user's histogram.

        Args:
            user_id: ID of the user making the query
            sequence: Sequence of item IDs

        Returns:
            is_attack: True if the user is identified as an attacker, False otherwise
        """
        if self.normal_histogram is None:
            raise ValueError(
                "Normal histogram not created. Call patched_create_normal_histogram() first."
            )

        if self.threshold is None:
            raise ValueError("Threshold not set. Call compute_threshold() first.")

        # Compute hardness degree
        hardness = self.compute_hardness_degree(sequence)

        # Initialize histogram for new users
        if user_id not in self.user_histograms:
            self.user_histograms[user_id] = np.zeros(self.num_bins + 1)

        # Update histogram
        self.user_histograms[user_id][hardness] += 1

        # Check if attack detection should be performed
        total_queries = np.sum(self.user_histograms[user_id])

        if total_queries >= 100:  # Minimum queries to detect attack
            # Normalize histogram
            normalized_hist = self.user_histograms[user_id] / total_queries

            # Compute distance
            distance = self.compute_pearson_distance(
                self.normal_histogram, normalized_hist
            )

            # Check if attack
            is_attack = distance > self.threshold

            return is_attack

        return False

    def detect_attack(self, user_id, query_sequences, visualize=False):
        """
        Detect if a user is performing a model extraction attack.

        Args:
            user_id: ID of the user to check
            query_sequences: List of query sequences from the user
            visualize: Whether to create and save a visualization

        Returns:
            is_attack: True if the user is identified as an attacker, False otherwise
            distance: Pearson distance between the user's histogram and the normal histogram
        """
        if self.normal_histogram is None:
            raise ValueError(
                "Normal histogram not created. Call patched_create_normal_histogram() first."
            )

        if self.threshold is None:
            raise ValueError("Threshold not set. Call compute_threshold() first.")

        # Reset user histogram
        self.user_histograms[user_id] = np.zeros(self.num_bins + 1)

        # Compute hardness degrees
        hardness_degrees = []
        for sequence in tqdm(query_sequences, desc="Computing hardness degrees"):
            hardness = self.compute_hardness_degree(sequence)
            hardness_degrees.append(hardness)
            self.user_histograms[user_id][hardness] += 1

        # Normalize histogram
        normalized_hist = self.user_histograms[user_id] / len(query_sequences)

        # Compute distance
        distance = self.compute_pearson_distance(self.normal_histogram, normalized_hist)

        # Check if attack
        is_attack = distance > self.threshold

        if visualize:
            self._visualize_histograms(user_id, normalized_hist, distance, is_attack)

        return is_attack, distance

    def _visualize_histograms(
        self, user_id, user_hist, distance, is_attack, output_dir="defense_results"
    ):
        """
        Create and save a visualization of the normal and user histograms.

        Args:
            user_id: ID of the user
            user_hist: User's histogram
            distance: Pearson distance
            is_attack: Whether the user is identified as an attacker
            output_dir: Directory to save the visualization
        """
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))

        # Plot histograms
        x = np.arange(self.num_bins + 1)
        width = 0.35

        plt.bar(x - width / 2, self.normal_histogram, width, label="Normal", alpha=0.7)
        plt.bar(x + width / 2, user_hist, width, label=f"User {user_id}", alpha=0.7)

        # Add labels and title
        plt.xlabel("Hardness Degree")
        plt.ylabel("Frequency")
        title = (
            f"Hardness Degree Histogram Comparison\nPearson Distance: {distance:.4f}"
        )
        if is_attack:
            title += " (ATTACK DETECTED)"
        plt.title(title)

        # Add threshold line
        plt.axhline(
            y=self.threshold,
            color="r",
            linestyle="-",
            label=f"Threshold: {self.threshold:.4f}",
        )

        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save figure
        plt.savefig(os.path.join(output_dir, f"histogram_user_{user_id}.png"))
        plt.close()

    def save(self, path="checkpoints/hoda_defense"):
        """Save HODA defense configuration and state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "normal_histogram": self.normal_histogram,
            "threshold": self.threshold,
            "num_bins": self.num_bins,
            "num_subclassifiers": self.num_subclassifiers,
        }

        torch.save(state, path)
        print(f"HODA defense state saved to {path}")

    def load(self, path="checkpoints/hoda_defense"):
        """Load HODA defense configuration and state"""
        try:
            state = torch.load(path, map_location=self.device)

            self.normal_histogram = state["normal_histogram"]
            self.threshold = state["threshold"]
            self.num_bins = state["num_bins"]
            self.num_subclassifiers = state["num_subclassifiers"]

            print(f"HODA defense state loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading HODA defense: {e}")
            return False
