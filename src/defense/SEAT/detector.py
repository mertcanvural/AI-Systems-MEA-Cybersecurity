import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union


class SEATDetector:
    """
    SEAT (Similarity Encoder by Adversarial Training) detector.

    Detects model extraction attacks by counting similar pairs of queries
    from the same account. Based on the paper:
    "SEAT: Similarity Encoder by Adversarial Training for Detecting
    Model Extraction Attack Queries"
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        similarity_threshold: Optional[float] = None,
        pairs_threshold: int = 50,
        fpr_target: float = 0.0001,
    ):
        """
        Initialize SEAT detector.

        Args:
            encoder: Trained similarity encoder
            similarity_threshold: L2 distance threshold for similar queries
                                  (calibrated automatically if None)
            pairs_threshold: Number of similar pairs needed to flag an account
            fpr_target: Target false positive rate for calibrating the threshold
        """
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.pairs_threshold = pairs_threshold
        self.fpr_target = fpr_target

        # Store queries and similar pairs by account
        self.account_queries = defaultdict(list)
        self.account_similar_pairs = defaultdict(int)
        self.flagged_accounts = set()

        # Performance optimization: store pair counts
        self.pair_counts = defaultdict(int)

    def calibrate_threshold(self, benign_queries: torch.Tensor) -> float:
        """
        Calibrate similarity threshold based on benign queries.

        Args:
            benign_queries: Tensor of benign query embeddings

        Returns:
            Calibrated similarity threshold
        """
        # Encode queries if they're raw embeddings
        with torch.no_grad():
            if not isinstance(benign_queries, torch.Tensor):
                benign_queries = torch.stack(benign_queries)

            # Get encodings if input isn't already encoded
            if len(benign_queries.shape) > 2:
                encoded_queries = self.encoder(benign_queries).cpu()
            else:
                encoded_queries = benign_queries

        # Compute pairwise distances
        benign_distances = []
        num_samples = min(1000, len(encoded_queries))

        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                dist = torch.norm(encoded_queries[i] - encoded_queries[j], p=2).item()
                benign_distances.append(dist)

        # Sort distances and find threshold for target FPR
        benign_distances.sort()
        total_pairs = len(benign_distances)

        # Make sure we have at least one FP (as in the paper)
        fpr_idx = max(1, int(self.fpr_target * total_pairs))
        self.similarity_threshold = benign_distances[fpr_idx]

        return self.similarity_threshold

    def process_query(self, query_embedding: torch.Tensor, account_id: str) -> bool:
        """
        Process a query and check if it's part of a model extraction attack.

        Args:
            query_embedding: Query embedding
            account_id: ID of the account making the query

        Returns:
            True if account is flagged as malicious, False otherwise
        """
        if account_id in self.flagged_accounts:
            return True

        if self.similarity_threshold is None:
            raise ValueError(
                "Similarity threshold not set. Run calibrate_threshold first."
            )

        # Encode query if needed
        with torch.no_grad():
            if len(query_embedding.shape) > 1 and query_embedding.shape[0] > 1:
                encoded_query = self.encoder(query_embedding).cpu().squeeze()
            else:
                encoded_query = query_embedding

        # Check similarity with all previous queries from this account
        for prev_query in self.account_queries[account_id]:
            dist = torch.norm(encoded_query - prev_query, p=2).item()

            if dist < self.similarity_threshold:
                # Found a similar pair
                self.account_similar_pairs[account_id] += 1

                # If threshold exceeded, flag the account
                if self.account_similar_pairs[account_id] >= self.pairs_threshold:
                    self.flagged_accounts.add(account_id)
                    return True

        # Store the query for future comparisons
        self.account_queries[account_id].append(encoded_query)
        return False

    def get_account_stats(self, account_id: str) -> Dict[str, Union[int, bool]]:
        """
        Get statistics for an account.

        Args:
            account_id: ID of the account

        Returns:
            Dictionary with account statistics
        """
        return {
            "queries": len(self.account_queries.get(account_id, [])),
            "similar_pairs": self.account_similar_pairs.get(account_id, 0),
            "is_flagged": account_id in self.flagged_accounts,
            "threshold": self.pairs_threshold,
        }

    def reset(self) -> None:
        """Reset the detector state."""
        self.account_queries = defaultdict(list)
        self.account_similar_pairs = defaultdict(int)
        self.flagged_accounts = set()


def compute_accounts_needed(detection_rate: float) -> int:
    """
    Calculate the number of accounts needed for attack based on detection rate.
    Formula from the SEAT paper.

    Args:
        detection_rate: Rate at which accounts are detected

    Returns:
        Number of accounts needed
    """
    if detection_rate >= 1.0:
        return 10  # Minimum value if detection rate is 100%

    return int(1 / (1 - detection_rate) + 1)
