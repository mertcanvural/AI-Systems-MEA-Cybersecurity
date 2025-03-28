import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from src.models.base_model import SimpleSequentialRecommender
from src.data.data_utils import load_movielens


class ModelExtractionAttack:
    """
    Implements a model extraction attack on sequential recommendation models.
    The attack queries the target model with synthetic sequences and uses the responses
    to train a surrogate model that mimics the target model's behavior.
    """

    def __init__(
        self,
        target_model_path,
        num_items,
        embedding_dim=256,
        device=None,
        query_budget=3000,
        top_k=10,
        margin=0.1,
        surrogate_model=None,
    ):
        """
        Initialize the model extraction attack.

        Args:
            target_model_path: Path to the target model checkpoint
            num_items: Number of items in the dataset
            embedding_dim: Embedding dimension for the surrogate model
            device: Device to use for inference (cuda or cpu)
            query_budget: Maximum number of queries to the target model
            top_k: Number of items in the recommendation list
            margin: Margin for the ranking loss
        """
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.query_budget = query_budget
        self.top_k = top_k
        self.margin = margin

        # Load target model
        self.target_model = self._load_target_model(
            target_model_path, num_items, embedding_dim
        )

        # Initialize surrogate model
        if surrogate_model is None:
            self.surrogate_model = SimpleSequentialRecommender(num_items, embedding_dim)
        else:
            self.surrogate_model = surrogate_model
        self.surrogate_model = self.surrogate_model.to(self.device)

        # Training data for surrogate model
        self.query_sequences = []
        self.recommendation_lists = []

    def _load_target_model(self, model_path, num_items, embedding_dim):
        """Load the target model from a checkpoint"""
        model = SimpleSequentialRecommender(num_items, embedding_dim)

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(
                    f"Loaded target model from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}"
                )
            else:
                model.load_state_dict(checkpoint)
                print("Loaded target model state dict")

            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            print(f"Error loading target model: {e}")
            raise e

    def _generate_query_sequence(self, strategy="autoregressive", seed_sequences=None):
        """
        Generate a sequence to query the target model.

        Args:
            strategy: Strategy to generate the query sequence
                     - "random": Generate a random sequence
                     - "autoregressive": Generate a sequence by querying the target model
            seed_sequences: Optional list of seed sequences to start from

        Returns:
            A sequence of item IDs
        """
        if strategy == "random":
            # Generate a random sequence of length between 1 and 10
            seq_length = np.random.randint(1, 10)
            sequence = np.random.randint(1, self.num_items, size=seq_length).tolist()
            return sequence

        elif strategy == "autoregressive":
            # Start with a random item or use a seed sequence if provided
            if seed_sequences and len(seed_sequences) > 0:
                # Randomly select a seed sequence
                sequence = np.random.choice(seed_sequences).copy()
            else:
                # Start with a random item
                sequence = [np.random.randint(1, self.num_items)]

            # Extend the sequence by querying the target model
            max_length = np.random.randint(
                3, 15
            )  # Random sequence length between 3 and 15

            for _ in range(max_length - len(sequence)):
                # Query the target model
                recommendations = self._query_target_model(sequence)

                # Sample from the top recommendations with probability bias towards higher ranks
                probs = np.linspace(0.5, 0.1, len(recommendations))
                probs = probs / probs.sum()
                next_item = np.random.choice(
                    [rec[0] for rec in recommendations], p=probs
                )

                # Add the item to the sequence
                sequence.append(next_item)

            return sequence
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _query_target_model(self, sequence):
        """
        Query the target model with a sequence and get recommendations.

        Args:
            sequence: A sequence of item IDs

        Returns:
            A list of (item_id, score) tuples representing the top-k recommendations
        """
        with torch.no_grad():
            # Convert sequence to tensor
            sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)

            # Get model predictions
            logits = self.target_model(sequence_tensor)

            # Get top-k predictions
            scores, indices = torch.topk(logits, k=self.top_k, dim=1)

            # Convert to list of (item_id, score) tuples
            recommendations = [
                (idx.item(), score.item()) for idx, score in zip(indices[0], scores[0])
            ]

        return recommendations

    def collect_data(self, strategy="autoregressive", seed_sequences=None):
        """
        Collect training data for the surrogate model by querying the target model.

        Args:
            strategy: Strategy to generate query sequences
            seed_sequences: Optional list of seed sequences to start from
        """
        print(
            f"Collecting data using {strategy} strategy, query budget: {self.query_budget}"
        )
        self.query_sequences = []
        self.recommendation_lists = []

        for _ in tqdm(range(self.query_budget), desc="Collecting data"):
            # Generate a query sequence
            sequence = self._generate_query_sequence(strategy, seed_sequences)

            # Query the target model
            recommendations = self._query_target_model(sequence)

            # Store the query and response
            self.query_sequences.append(sequence)
            self.recommendation_lists.append(recommendations)

        print(f"Collected {len(self.query_sequences)} query-response pairs")

    def _calculate_surrogate_loss(self, sequence, recommendations):
        """
        Calculate the loss for training the surrogate model.

        Args:
            sequence: A sequence of item IDs
            recommendations: A list of (item_id, score) tuples representing the top-k recommendations

        Returns:
            Loss value
        """
        # Convert sequence to tensor
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)

        # Get surrogate model predictions
        logits = self.surrogate_model(sequence_tensor)[0]  # Remove batch dimension

        # Extract item IDs and their ranking positions
        ranked_items = [item_id for item_id, _ in recommendations]

        # Calculate pairwise ranking loss
        # For each pair of adjacent items in the ranking, the higher-ranked item should have a higher score
        ranking_loss = 0.0
        for i in range(len(ranked_items) - 1):
            item_i = ranked_items[i]
            item_j = ranked_items[i + 1]
            score_i = logits[item_i]
            score_j = logits[item_j]
            # Higher ranked item should have higher score
            ranking_loss += torch.max(
                torch.tensor(0.0).to(self.device), score_j - score_i + self.margin
            )

        # Negative sampling - items in the top-k should have higher scores than random items
        negative_samples = []
        while len(negative_samples) < len(ranked_items):
            neg_item = np.random.randint(1, self.num_items)
            if neg_item not in ranked_items and neg_item not in negative_samples:
                negative_samples.append(neg_item)

        negative_loss = 0.0
        for i in range(len(ranked_items)):
            pos_item = ranked_items[i]
            neg_item = negative_samples[i]
            pos_score = logits[pos_item]
            neg_score = logits[neg_item]
            # Positive item should have higher score than negative item
            negative_loss += torch.max(
                torch.tensor(0.0).to(self.device), neg_score - pos_score + self.margin
            )

        return ranking_loss + negative_loss

    def train_surrogate_model(self, num_epochs=30, learning_rate=0.001, batch_size=32):
        """
        Train the surrogate model using the collected data.

        Args:
            num_epochs: Number of epochs for training
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training
        """
        if len(self.query_sequences) == 0:
            print("No data collected. Please call collect_data() first.")
            return

        print(f"Training surrogate model for {num_epochs} epochs")
        self.surrogate_model.train()

        # Prepare optimizer
        optimizer = torch.optim.Adam(
            self.surrogate_model.parameters(), lr=learning_rate
        )

        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Create random batches
            indices = np.arange(len(self.query_sequences))
            np.random.shuffle(indices)

            # Process in batches
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                batch_loss = 0.0

                # Process each sequence in the batch
                for idx in batch_indices:
                    sequence = self.query_sequences[idx]
                    recommendations = self.recommendation_lists[idx]

                    # Calculate loss
                    loss = self._calculate_surrogate_loss(sequence, recommendations)
                    batch_loss += loss

                # Update weights
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(indices):.4f}")

        # Set to evaluation mode after training
        self.surrogate_model.eval()
        print("Surrogate model training complete")

    def save_surrogate_model(self, save_path="checkpoints/surrogate_model.pt"):
        """Save the trained surrogate model"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.surrogate_model.state_dict(), save_path)
        print(f"Surrogate model saved to {save_path}")

    def evaluate_attack(self, test_sequences=None, k_values=[1, 5, 10, 20]):
        """
        Evaluate how well the surrogate model mimics the target model.

        Args:
            test_sequences: List of test sequences. If None, generates random sequences.
            k_values: List of k values for evaluation

        Returns:
            Dictionary of metric values
        """
        if test_sequences is None:
            # Generate random test sequences
            test_sequences = []
            for _ in range(100):
                test_sequences.append(self._generate_query_sequence(strategy="random"))

        metrics = {}

        # Lists to store recommendation overlaps
        overlaps = defaultdict(list)
        rank_correlations = []

        self.target_model.eval()
        self.surrogate_model.eval()

        with torch.no_grad():
            for sequence in tqdm(test_sequences, desc="Evaluating attack"):
                # Get target model recommendations
                target_recs = self._query_target_model(sequence)
                target_items = [item for item, _ in target_recs]

                # Get surrogate model recommendations
                sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(
                    self.device
                )
                surrogate_logits = self.surrogate_model(sequence_tensor)
                scores, indices = torch.topk(surrogate_logits, k=max(k_values), dim=1)
                surrogate_items = indices[0].tolist()

                # Calculate overlap at different k values
                for k in k_values:
                    target_at_k = set(target_items[:k])
                    surrogate_at_k = set(surrogate_items[:k])
                    overlap = len(target_at_k.intersection(surrogate_at_k)) / k
                    overlaps[k].append(overlap)

                # Calculate Spearman rank correlation between the two rankings
                # First create dictionaries mapping item to rank
                target_ranks = {item: i for i, item in enumerate(target_items)}
                surrogate_ranks = {item: i for i, item in enumerate(surrogate_items)}

                # Find common items
                common_items = set(target_items).intersection(
                    set(surrogate_items[: len(target_items)])
                )

                if common_items:
                    # Calculate differences in ranks
                    rank_diffs = [
                        (target_ranks[item] - surrogate_ranks[item]) ** 2
                        for item in common_items
                    ]
                    n = len(common_items)

                    # Spearman correlation = 1 - (6 * sum(rank_diffs) / (n * (n^2 - 1)))
                    if n > 1:  # Need at least 2 items to calculate correlation
                        correlation = 1 - (6 * sum(rank_diffs) / (n * (n**2 - 1)))
                        rank_correlations.append(correlation)

        # Calculate average metrics
        for k in k_values:
            metrics[f"overlap@{k}"] = np.mean(overlaps[k])

        if rank_correlations:
            metrics["rank_correlation"] = np.mean(rank_correlations)

        # Print metrics
        print("Attack evaluation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Model Extraction Attack on Recommendation System"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to target model checkpoint",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/surrogate_model.pt",
        help="Path to save the surrogate model",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension for the models",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=3000,
        help="Number of queries to the target model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs for the surrogate model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["random", "autoregressive"],
        default="autoregressive",
        help="Strategy for generating query sequences",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ml-1m/ratings.dat",
        help="Path to dataset for loading movie IDs",
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Load basic dataset info to get num_items
    data = load_movielens(args.data_path)
    num_items = data["num_items"]
    print(f"Dataset has {num_items} items")

    # Create attack instance
    attack = ModelExtractionAttack(
        target_model_path=args.target_model,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        query_budget=args.queries,
    )

    # Collect data from the target model
    attack.collect_data(strategy=args.strategy)

    # Train the surrogate model
    attack.train_surrogate_model(num_epochs=args.epochs, batch_size=args.batch_size)

    # Evaluate the attack
    attack.evaluate_attack()

    # Save the surrogate model
    attack.save_surrogate_model(args.save_path)

    print(
        f"Model extraction attack completed. Surrogate model saved to {args.save_path}"
    )


if __name__ == "__main__":
    main()
