import os
import torch
import numpy as np
from tqdm import tqdm
from src.models.evaluation import evaluate_model

class Trainer:
    """Trainer class for recommender models"""
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        device=None,
        checkpoint_dir="checkpoints"
    ):
        """
        Initialize trainer
        
        args:
            model: model to train
            train_dataloader: dataloader for training data
            val_dataloader: dataloader for validation data
            optimizer: optimizer for training
            device: device to use for training
            checkpoint_dir: directory to save checkpoints
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_dir = checkpoint_dir
        
        # Move model to device
        self.model.to(self.device)
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self, num_epochs, patience=5):
        """
        Train model
        
        args:
            num_epochs: number of epochs to train
            patience: number of epochs to wait for validation improvement
        
        returns:
            model: trained model
            results: dictionary with training results
        """
        best_val_metric = 0
        best_epoch = 0
        patience_counter = 0
        
        # Store metrics
        results = {
            "train_loss": [],
            "val_metrics": []
        }
        
        for epoch in range(num_epochs):
            # Train epoch
            train_loss = self._train_epoch()
            results["train_loss"].append(train_loss)
            
            # Evaluate on validation set
            val_metrics = evaluate_model(self.model, self.val_dataloader)
            results["val_metrics"].append(val_metrics)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            for metric_name, metric_value in val_metrics.items():
                print(f"  Val {metric_name}: {metric_value:.4f}")
            
            # Check for improvement
            current_val_metric = val_metrics.get("hit_rate_at_k_10", 0)
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                self._save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
                
            # Save latest model
            self._save_checkpoint("latest_model.pt")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        print(f"Best validation metric: {best_val_metric:.4f} at epoch {best_epoch+1}")
        
        # Load best model
        self._load_checkpoint("best_model.pt")
        
        return self.model, results
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.train_dataloader, desc="Training"):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Calculate loss
            loss = self.model.get_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss
        epoch_loss /= num_batches
        
        return epoch_loss
    
    def _save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
    
    def _load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])