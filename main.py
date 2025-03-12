import os
import argparse
import random
import numpy as np
import torch
from src.data.data_utils import (
    load_movielens,
    create_train_val_test_splits,
    create_data_loaders,
)
from src.models.base_model import SimpleSequentialRecommender
from src.models.evaluation import evaluate_model
from src.utils.trainer import Trainer


def set_seed(seed):
    """set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="gro defense for recommender systems")

    # data arguments
    parser.add_argument(
        "--data_dir", type=str, default="data", help="directory containing the datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
        choices=["ml-1m", "ml-20m"],
        help="dataset to use",
    )
    parser.add_argument(
        "--min_seq_length", type=int, default=5, help="minimum sequence length to keep"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=50,
        help="maximum sequence length after padding/truncation",
    )

    # model arguments
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="dimension of item embeddings"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="dropout rate")

    # training arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="number of epochs to train"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="patience for early stopping"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    # set random seed
    set_seed(args.seed)

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # load dataset
    data_path = os.path.join(args.data_dir, f"{args.dataset}.csv")
    data = load_movielens(data_path, min_sequence_length=args.min_seq_length)

    # create train/val/test splits
    splits = create_train_val_test_splits(data["user_sequences"])

    # create dataloaders
    dataloaders = create_data_loaders(
        splits, batch_size=args.batch_size, max_seq_length=args.max_seq_length
    )

    print("dataset statistics:")
    print(f"  number of items: {data['num_items']}")
    print(f"  number of training sequences: {len(splits['train_sequences'])}")
    print(f"  number of validation sequences: {len(splits['val_sequences'])}")
    print(f"  number of test sequences: {len(splits['test_sequences'])}")

    # create model
    model = SimpleSequentialRecommender(
        num_items=data["num_items"],
        embedding_dim=args.embedding_dim,
        dropout_rate=args.dropout_rate,
    )

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # train model
    print("training model...")
    model, training_results = trainer.train(
        num_epochs=args.num_epochs, patience=args.patience
    )

    # evaluate on test set
    print("evaluating on test set...")
    test_results = evaluate_model(model, dataloaders["test"])

    # print results
    print("test results:")
    for metric_name, metric_value in test_results.items():
        print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
