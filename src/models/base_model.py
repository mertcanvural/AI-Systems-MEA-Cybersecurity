import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def calculate_mrr(predictions, targets):
    """
    Calculate Mean Reciprocal Rank (MRR)

    Args:
        predictions: tensor of predicted item rankings
        targets: tensor of actual target items

    Returns:
        mrr: mean reciprocal rank score
    """
    # Get the rank of the true item in the predictions
    _, indices = torch.sort(predictions, dim=1, descending=True)
    ranks = []

    for i, target in enumerate(targets):
        # Find the position (rank) of the target in the predictions
        # Add 1 because ranks start from 1, not 0
        rank = (indices[i] == target).nonzero().item() + 1
        ranks.append(1.0 / rank)  # Reciprocal rank

    # Mean of the reciprocal ranks
    return torch.tensor(ranks).mean().item()


def calculate_ndcg(predictions, targets, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)

    Args:
        predictions: tensor of predicted item scores
        targets: tensor of actual target items
        k: number of items to consider (usually 10)

    Returns:
        ndcg: NDCG score
    """
    # Get top-k predictions
    _, indices = torch.topk(predictions, k=k, dim=1)

    ndcg_scores = []
    for i, target in enumerate(targets):
        # Check if target is in top-k predictions
        if target in indices[i]:
            # Find position of target in top-k
            rank = (indices[i] == target).nonzero().item() + 1
            # Calculate DCG: relevant items are scored 1
            dcg = 1.0 / torch.log2(torch.tensor(rank + 1).float())
            # IDCG is 1 for a single relevant item at position 1
            idcg = 1.0
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return torch.tensor(ndcg_scores).mean().item()


class RecommenderBase(nn.Module):
    def __init__(self, num_items, embedding_dim=64):
        """
        num_items     : number of items in the dataset
        embedding_dim : dimension of item embeddings
        """
        super(RecommenderBase, self).__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # item embeddings
        self.item_embeddings = nn.Embedding(num_items, embedding_dim, padding_idx=0)

    def forward(self, x):
        """
        parent forward pass
            x      : input tensor
            logits : prediction logits
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_loss(self, batch):
        """
        Calculate loss for a batch

        args:
            batch: dictionary with input_ids and labels

        returns:
            loss: loss value
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        logits = self.forward(input_ids)
        loss = F.cross_entropy(logits, labels)

        return loss

    def predict_next_item(self, sequence):
        """
        Predict the next item for a sequence

        args:
            sequence: list or tensor of item ids

        returns:
            scores: scores for each item
        """
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.long)

        sequence = sequence.unsqueeze(0).to(self.device)  # add batch dimension
        with torch.no_grad():
            logits = self.forward(sequence)

        return logits.squeeze(0)

    def get_item_embeddings(self):
        """
        Get item embeddings

        returns:
            embeddings: item embedding matrix
        """
        return self.item_embeddings.weight.detach().cpu().numpy()

    def evaluate_metrics(self, dataloader):
        """
        Evaluate model on multiple metrics

        args:
            dataloader: dataloader with evaluation data

        returns:
            metrics: dictionary with evaluation metrics
        """
        self.eval()
        total_loss = 0
        hits_10 = 0
        mrr_sum = 0
        ndcg_sum = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                logits = self.forward(input_ids)
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()

                # Hit@10
                _, top_indices = torch.topk(logits, k=10, dim=1)
                for i, label in enumerate(labels):
                    if label in top_indices[i]:
                        hits_10 += 1
                    total += 1

                # MRR and NDCG
                mrr_sum += calculate_mrr(logits, labels) * len(labels)
                ndcg_sum += calculate_ndcg(logits, labels) * len(labels)

        metrics = {
            "loss": total_loss / len(dataloader),
            "hit@10": hits_10 / total if total > 0 else 0,
            "mrr": mrr_sum / total if total > 0 else 0,
            "ndcg@10": ndcg_sum / total if total > 0 else 0,
        }

        return metrics

    @property
    def device(self):
        """Get model device"""
        return next(self.parameters()).device

    def save(self, path):
        """Save model to path"""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from path"""
        self.load_state_dict(torch.load(path))


class SimpleSequentialRecommender(RecommenderBase):
    """Simple sequential recommender model based on average pooling"""

    def __init__(self, num_items, embedding_dim=64, dropout_rate=0.1):
        """
        Initialize simple sequential recommender

        args:
            num_items: number of items in the dataset
            embedding_dim: dimension of item embeddings
            dropout_rate: dropout rate
        """
        super(SimpleSequentialRecommender, self).__init__(num_items, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim, num_items)

        # Initialize weights
        nn.init.xavier_normal_(self.item_embeddings.weight)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        Forward pass

        args:
            x: input tensor of shape (batch_size, seq_length)

        returns:
            logits: prediction logits of shape (batch_size, num_items)
        """
        # Create mask for padding
        mask = (x > 0).float().unsqueeze(-1)  # (batch_size, seq_length, 1)

        # Get item embeddings
        item_emb = self.item_embeddings(x)  # (batch_size, seq_length, embedding_dim)

        # Apply mask to ignore padding
        masked_item_emb = item_emb * mask

        # Average pooling
        sum_emb = masked_item_emb.sum(dim=1)  # (batch_size, embedding_dim)
        seq_length = mask.sum(dim=1)  # (batch_size, 1)
        seq_length = torch.clamp(seq_length, min=1.0)  # avoid division by zero
        mean_emb = sum_emb / seq_length

        # Apply dropout
        mean_emb = self.dropout(mean_emb)

        # Predict next item
        logits = self.fc(mean_emb)  # (batch_size, num_items)

        return logits

    def get_similar_items(self, item_ids, k=10, genre_bias=None):
        """
        Get similar items based on embedding similarity

        args:
            item_ids: list of item ids to find similar items for
            k: number of similar items to return
            genre_bias: optional dictionary mapping genre -> weight for boosting specific genres

        returns:
            similar_items: list of tuples (item_id, similarity_score)
        """
        if not isinstance(item_ids, list):
            item_ids = [item_ids]

        # Get embeddings
        with torch.no_grad():
            item_embeddings = self.item_embeddings.weight

            # Calculate average embedding for query items
            query_embedding = torch.mean(
                torch.stack([item_embeddings[item_id] for item_id in item_ids]), dim=0
            )

            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(
                query_embedding.unsqueeze(0), item_embeddings, dim=1
            )

            # Apply genre bias if provided
            if genre_bias is not None and hasattr(self, "item_to_genre"):
                for item_id in range(self.num_items):
                    if item_id in self.item_to_genre:
                        genres = self.item_to_genre[item_id]
                        boost = 0
                        for genre in genres:
                            if genre in genre_bias:
                                boost += genre_bias[genre]
                        cos_sim[item_id] += boost

            # Remove original items
            for item_id in item_ids:
                cos_sim[item_id] = -1.0

            # Get top-k items
            topk_scores, topk_indices = torch.topk(cos_sim, k=k)

            similar_items = [
                (idx.item(), score.item())
                for idx, score in zip(topk_indices, topk_scores)
            ]

        return similar_items

    def recommend_with_genre_bias(self, sequence, k=10, genre_weights=None):
        """
        Enhanced recommendation method with genre bias

        args:
            sequence: list of item ids
            k: number of recommendations
            genre_weights: dict of genre -> weight bias

        returns:
            recommendations: list of tuples (item_id, score)
        """
        # Get model predictions
        model_scores = self.predict_next_item(sequence)

        # Get embedding similarity
        similarity_scores = self.get_similar_items(
            sequence, k=self.num_items, genre_bias=genre_weights
        )
        sim_dict = dict(similarity_scores)

        # Combine scores (50% model, 50% similarity)
        combined_scores = torch.zeros_like(model_scores)
        for i in range(len(combined_scores)):
            model_score = model_scores[i].item()
            sim_score = sim_dict.get(i, 0)
            combined_scores[i] = 0.5 * model_score + 0.5 * sim_score

        # Get top-k items
        topk_scores, topk_indices = torch.topk(combined_scores, k=k)
        recommendations = [
            (idx.item(), score.item()) for idx, score in zip(topk_indices, topk_scores)
        ]

        return recommendations

    def set_genre_mapping(self, item_to_genre):
        """Add genre information to the model for improved recommendations"""
        self.item_to_genre = item_to_genre
