import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

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
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
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