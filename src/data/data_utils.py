import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SequentialRecommendationDataset(Dataset):
    """dataset for sequential recommendation"""
    
    def __init__(self, sequences, targets, max_seq_length=50):
        """
        initialize dataset
        
        args:
            sequences: list of item sequences
            targets: list of target items
            max_seq_length: maximum sequence length after padding/truncation
        """
        self.sequences = sequences
        self.targets = targets
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        # truncate or pad sequence
        if len(sequence) > self.max_seq_length:
            # truncate to max_seq_length (keep most recent)
            sequence = sequence[-self.max_seq_length:]
        elif len(sequence) < self.max_seq_length:
            # pad with zeros at the beginning
            sequence = [0] * (self.max_seq_length - len(sequence)) + sequence
        
        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'labels': torch.tensor(target, dtype=torch.long)
        }
        
def load_movielens(file_path, min_sequence_length=5):
    """
    load movielens dataset and convert to sequences
    
    args:
        file_path: path to movielens csv file
        min_sequence_length: minimum number of interactions to keep a user
        
    returns:
        data: dictionary with user_sequences and num_items
    """
    print(f"loading movielens dataset from {file_path}")
    
    # read csv file
    df = pd.read_csv(file_path)
    
    # sort by user and timestamp
    df = df.sort_values(['userId', 'timestamp'])
    
    # create user sequences
    user_sequences = {}
    
    print("creating user sequences:")
    for user_id, group in tqdm(df.groupby('userId')):
        item_ids = group['movieId'].values.tolist()
        
        # only keep users with minimum sequence length
        if len(item_ids) >= min_sequence_length:
            user_sequences[user_id] = item_ids
    
    print(f"created {len(user_sequences)} user sequences")
    
    # get all unique items
    all_items = set()
    for items in user_sequences.values():
        all_items.update(items)
    
    num_items = max(all_items) + 1  # add 1 for padding/unknown
    
    return {
        'user_sequences': user_sequences,
        'num_items': num_items
    }

def create_train_val_test_splits(user_sequences, val_ratio=0.1, test_ratio=0.1):
    """
    create train/validation/test splits from user sequences
    
    args:
        user_sequences: dictionary mapping user_id to item sequence
        val_ratio: ratio of users to use for validation
        test_ratio: ratio of users to use for testing
        
    returns:
        splits: dictionary with train/val/test sequences and targets
    """
    print("creating dataset splits...")
    
    # get all users
    users = list(user_sequences.keys())
    np.random.shuffle(users)
    
    # determine split points
    num_users = len(users)
    num_val = int(num_users * val_ratio)
    num_test = int(num_users * test_ratio)
    
    # split users
    val_users = users[:num_val]
    test_users = users[num_val:num_val+num_test]
    train_users = users[num_val+num_test:]
    
    # function to create input/target pairs
    def create_sequences_and_targets(user_subset):
        sequences = []
        targets = []
        
        for user_id in user_subset:
            sequence = user_sequences[user_id]
            
            # use last item as target, rest as input
            if len(sequence) > 1:
                sequences.append(sequence[:-1])
                targets.append(sequence[-1])
        
        return sequences, targets
    
    # create splits
    train_sequences, train_targets = create_sequences_and_targets(train_users)
    val_sequences, val_targets = create_sequences_and_targets(val_users)
    test_sequences, test_targets = create_sequences_and_targets(test_users)
    
    return {
        'train_sequences': train_sequences,
        'train_targets': train_targets,
        'val_sequences': val_sequences,
        'val_targets': val_targets,
        'test_sequences': test_sequences,
        'test_targets': test_targets
    }

def create_data_loaders(splits, batch_size=32, max_seq_length=50):
    """
    create dataloader objects for all splits
    
    args:
        splits: dictionary with train/val/test sequences and targets
        batch_size: batch size for dataloader
        max_seq_length: maximum sequence length
        
    returns:
        dataloaders: dictionary with train/val/test dataloaders
    """
    # create datasets
    train_dataset = SequentialRecommendationDataset(
        splits['train_sequences'], splits['train_targets'], 
        max_seq_length=max_seq_length
    )
    
    val_dataset = SequentialRecommendationDataset(
        splits['val_sequences'], splits['val_targets'], 
        max_seq_length=max_seq_length
    )
    
    test_dataset = SequentialRecommendationDataset(
        splits['test_sequences'], splits['test_targets'], 
        max_seq_length=max_seq_length
    )
    
    # create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }