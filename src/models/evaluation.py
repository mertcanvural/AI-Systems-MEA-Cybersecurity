import torch
import numpy as np
from tqdm import tqdm

def hit_rate_at_k(predictions, targets, k=10):
    """
    Calculate hit rate at k
    
    args:
        predictions: predicted scores or ranks, shape (batch_size, num_items)
        targets: target items, shape (batch_size,)
        k: number of items to consider
        
    returns:
        hit_rate: hit rate at k
    """
    # Get top k predictions
    _, top_k_indices = torch.topk(predictions, k, dim=1)
    
    # Check if target is in top k
    targets = targets.unsqueeze(1)  # (batch_size, 1)
    hit = (top_k_indices == targets).any(dim=1)
    
    # Calculate hit rate
    hit_rate = hit.float().mean().item()
    
    return hit_rate

def ndcg_at_k(predictions, targets, k=10):
    """
    Calculate normalized discounted cumulative gain at k
    
    args:
        predictions: predicted scores or ranks, shape (batch_size, num_items)
        targets: target items, shape (batch_size,)
        k: number of items to consider
        
    returns:
        ndcg: normalized discounted cumulative gain at k
    """
    # Get top k predictions
    _, top_k_indices = torch.topk(predictions, k, dim=1)
    
    # Create tensor of size (batch_size, k) filled with targets
    targets = targets.unsqueeze(1).expand(-1, k)
    
    # Check where in the top k the target appears
    hits = (top_k_indices == targets)
    
    # Calculate position of hit (1-indexed)
    positions = torch.arange(1, k+1, dtype=torch.float, device=predictions.device)
    positions = positions.unsqueeze(0).expand_as(hits)
    
    # Calculate DCG
    dcg = hits.float() / torch.log2(positions + 1)
    dcg = dcg.sum(dim=1)
    
    # Calculate ideal DCG (target at position 1)
    idcg = torch.tensor(1.0, device=predictions.device) / torch.log2(torch.tensor(2.0, device=predictions.device))
    
    # Calculate NDCG
    ndcg = dcg / idcg
    
    # Average over batch
    ndcg = ndcg.mean().item()
    
    return ndcg

def evaluate_model(model, dataloader, metrics=None, k_values=None):
    """
    Evaluate model on dataloader
    
    args:
        model: model to evaluate
        dataloader: dataloader with evaluation data
        metrics: list of metric functions to use
        k_values: list of k values for top-k metrics
        
    returns:
        results: dictionary with evaluation results
    """
    if metrics is None:
        metrics = [hit_rate_at_k, ndcg_at_k]
    if k_values is None:
        k_values = [5, 10, 20]
    
    device = model.device
    model.eval()
    
    results = {}
    for k in k_values:
        for metric_fn in metrics:
            metric_name = f"{metric_fn.__name__}_{k}"
            results[metric_name] = 0.0
    
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate metrics
            for k in k_values:
                for metric_fn in metrics:
                    metric_name = f"{metric_fn.__name__}_{k}"
                    metric_value = metric_fn(logits, labels, k=k)
                    results[metric_name] += metric_value
            
            num_batches += 1
    
    # Average metrics
    for k in k_values:
        for metric_fn in metrics:
            metric_name = f"{metric_fn.__name__}_{k}"
            results[metric_name] /= num_batches
    
    return results