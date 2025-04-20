import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.base_model import SimpleSequentialRecommender
from src.defense.hoda_defense import HODADefense
from src.attack.model_extraction import ModelExtractionAttack
from src.data.data_utils import load_movielens


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Apply HODA Defense Against Model Extraction Attacks"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to target model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ml-1m/ratings.dat",
        help="Path to dataset",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension for models",
    )
    parser.add_argument(
        "--num-subclassifiers",
        type=int,
        default=5,
        help="Number of subclassifiers to use (5 or 11 recommended)",
    )
    parser.add_argument(
        "--train-subclassifiers",
        action="store_true",
        help="Train subclassifiers from scratch",
    )
    parser.add_argument(
        "--training-epochs",
        type=int,
        default=100,
        help="Number of epochs for training subclassifiers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--attack-queries",
        type=int,
        default=1000,
        help="Number of queries for model extraction attack",
    )
    parser.add_argument(
        "--attack-epochs",
        type=int,
        default=10,
        help="Number of epochs for training surrogate models",
    )
    parser.add_argument(
        "--detection-num-seqs",
        type=int,
        default=100,
        help="Number of sequences to use for attack detection",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="defense_results",
        help="Directory to save results",
    )
    return parser.parse_args()


def prepare_data(data_path, batch_size):
    """Prepare data for training and evaluation"""
    print("Loading and preparing data...")

    # Load data
    data = load_movielens(data_path)
    num_items = data["num_items"]

    # Create datasets
    train_seqs = []
    train_targets = []
    val_seqs = []
    val_targets = []
    test_seqs = []
    test_targets = []

    for user_id, sequence in data["user_sequences"].items():
        if len(sequence) < 3:
            continue

        # Last item is test, second-to-last is validation
        test_item = sequence[-1]
        val_item = sequence[-2]
        train_seq = sequence[:-2]

        # Skip if train sequence is empty
        if len(train_seq) == 0:
            continue

        train_seqs.append(train_seq)
        train_targets.append(val_item)

        val_seqs.append(sequence[:-1])
        val_targets.append(test_item)

        test_seqs.append(sequence)
        test_targets.append(0)  # Placeholder, not used

    # Create data loaders
    def pad_sequences(sequences):
        max_len = max(len(seq) for seq in sequences)
        padded = np.zeros((len(sequences), max_len), dtype=np.int64)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq
        return padded

    train_seqs_padded = pad_sequences(train_seqs)
    train_targets = np.array(train_targets, dtype=np.int64)

    val_seqs_padded = pad_sequences(val_seqs)
    val_targets = np.array(val_targets, dtype=np.int64)

    test_seqs_padded = pad_sequences(test_seqs)
    test_targets = np.array(test_targets, dtype=np.int64)

    train_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(train_seqs_padded), torch.LongTensor(train_targets)
    )

    val_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(val_seqs_padded), torch.LongTensor(val_targets)
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(test_seqs_padded), torch.LongTensor(test_targets)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset has {num_items} items")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_items": num_items,
        "train_seqs": train_seqs,
        "val_seqs": val_seqs,
        "test_seqs": test_seqs,
    }


def load_target_model(model_path, num_items, embedding_dim, device):
    """Load target model from checkpoint"""
    model = SimpleSequentialRecommender(num_items, embedding_dim)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded target model from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}"
            )
        else:
            model.load_state_dict(checkpoint)
            print("Loaded target model state dict")

        model = model.to(device)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading target model: {e}")
        raise e


def attack_model(
    target_model, num_items, embedding_dim, device, query_budget=1000, attack_epochs=10
):
    """Apply model extraction attack on the target model"""
    print(f"Applying model extraction attack with {query_budget} queries...")

    # Create a temporary file path but don't use it
    # The target model will be set directly instead
    attack = ModelExtractionAttack(
        # Pass a dummy path that won't be used
        target_model_path="dummy_path",
        num_items=num_items,
        embedding_dim=embedding_dim,
        device=device,
        query_budget=query_budget,
        top_k=10,
        # Pass the model directly as surrogate_model to avoid loading from path
        surrogate_model=SimpleSequentialRecommender(num_items, embedding_dim),
    )

    # Set the target model directly - this overrides the _load_target_model call
    attack.target_model = target_model

    # Collect data through queries
    attack.collect_data(strategy="autoregressive")

    # Train surrogate model
    attack.train_surrogate_model(num_epochs=attack_epochs)

    # Evaluate attack
    attack_metrics = attack.evaluate_attack()

    # Get query sequences
    query_sequences = attack.query_sequences

    return attack.surrogate_model, attack_metrics, query_sequences


def evaluate_model(model, data_loader, device, top_k=10):
    """Evaluate model performance"""
    model.eval()

    hits = 0
    ndcg = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            sequences, targets = batch

            sequences = sequences.to(device)
            targets = targets.to(device)

            # Get predictions
            logits = model(sequences)

            # Get top-k recommendations
            _, indices = torch.topk(logits, k=top_k, dim=1)

            # Calculate metrics
            for i, target in enumerate(targets):
                total += 1
                if target.item() in indices[i]:
                    hits += 1
                    rank = torch.where(indices[i] == target.item())[0][0].item() + 1
                    ndcg += 1.0 / np.log2(rank + 1)

    hr = hits / total if total > 0 else 0
    ndcg = ndcg / total if total > 0 else 0

    return hr, ndcg


def apply_hoda_defense_and_evaluate(args):
    """Apply HODA defense and evaluate its effectiveness"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    data = prepare_data(args.data_path, args.batch_size)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    num_items = data["num_items"]
    train_seqs = data["train_seqs"]
    val_seqs = data["val_seqs"]

    # Load target model
    target_model = load_target_model(
        args.target_model, num_items, args.embedding_dim, device
    )

    # Evaluate original model
    print("Evaluating original model...")
    original_hr, original_ndcg = evaluate_model(target_model, val_loader, device)
    print(f"Original model - HR@10: {original_hr:.4f}, NDCG@10: {original_ndcg:.4f}")

    # Setup HODA defense
    os.makedirs(args.output_dir, exist_ok=True)
    hoda = HODADefense(
        target_model=target_model,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        device=device,
        num_subclassifiers=args.num_subclassifiers,
    )

    # Train or load subclassifiers
    subclassifiers_dir = os.path.join(args.output_dir, "subclassifiers")
    if args.train_subclassifiers:
        # Train subclassifiers from scratch
        print("Training subclassifiers from scratch...")
        hoda.train_subclassifiers(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.training_epochs,
            save_dir=subclassifiers_dir,
        )
    else:
        # Try to load pre-trained subclassifiers
        print("Loading pre-trained subclassifiers...")
        try:
            hoda.load_subclassifiers(checkpoint_dir=subclassifiers_dir)
        except Exception as e:
            print(f"Error loading subclassifiers: {e}")
            print("Training subclassifiers instead...")
            hoda.train_subclassifiers(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.training_epochs,
                save_dir=subclassifiers_dir,
            )

    # Create normal histogram
    print("Creating normal histogram...")
    normal_sequences = (
        train_seqs + val_seqs
    )  # Use both training and validation sequences
    hoda.patched_create_normal_histogram(normal_sequences)

    # Compute detection threshold
    print("Computing detection threshold...")
    hoda.compute_threshold(normal_sequences, numseq=1000, nums=args.detection_num_seqs)

    # Save HODA defense state
    hoda.save(os.path.join(args.output_dir, "hoda_defense.pt"))

    # Attack original model
    print("Attacking original model...")
    original_surrogate, original_attack_metrics, attack_sequences = attack_model(
        target_model,
        num_items,
        args.embedding_dim,
        device,
        args.attack_queries,
        args.attack_epochs,
    )

    # Evaluate surrogate model from attack on original model
    print("Evaluating surrogate model (attack on original)...")
    original_surrogate_hr, original_surrogate_ndcg = evaluate_model(
        original_surrogate, val_loader, device
    )
    print(
        f"Surrogate of original - HR@10: {original_surrogate_hr:.4f}, NDCG@10: {original_surrogate_ndcg:.4f}"
    )

    # Detect attack
    print("Detecting attack...")
    is_attack, distance = hoda.detect_attack(
        user_id="attacker",
        query_sequences=attack_sequences[: args.detection_num_seqs],
        visualize=True,
    )

    print(f"Attack detected: {is_attack}")
    print(f"Pearson distance: {distance:.4f}")

    # Create benign user scenario
    print("Simulating benign user...")
    # Select indices then access the sequences
    indices = np.random.choice(
        len(normal_sequences), size=args.detection_num_seqs, replace=True
    )
    benign_sequences = [normal_sequences[i] for i in indices]
    is_benign_attack, benign_distance = hoda.detect_attack(
        user_id="benign",
        query_sequences=benign_sequences,
        visualize=True,
    )

    print(f"Benign user detected as attack: {is_benign_attack}")
    print(f"Benign user Pearson distance: {benign_distance:.4f}")

    # Compute attack success rate
    original_attack_success = (
        original_surrogate_hr / original_hr if original_hr > 0 else 0
    )

    # Print summary
    print("\n" + "=" * 50)
    print("HODA DEFENSE EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Original model HR@10: {original_hr:.4f}, NDCG@10: {original_ndcg:.4f}")
    print(
        f"Surrogate of original HR@10: {original_surrogate_hr:.4f}, NDCG@10: {original_surrogate_ndcg:.4f}"
    )
    print(f"Attack success rate: {original_attack_success:.2%}")
    print("-" * 50)
    print(f"Detection threshold: {hoda.threshold:.4f}")
    print(f"Attack distance: {distance:.4f}")
    print(f"Benign user distance: {benign_distance:.4f}")
    print(f"Attack correctly detected: {is_attack}")
    print(f"Benign user correctly identified: {not is_benign_attack}")
    print(f"False positive rate on benign user: {int(is_benign_attack) * 100:.0f}%")
    print(f"Number of queries monitored: {args.detection_num_seqs}")
    print("=" * 50)

    # Save metrics to file
    metrics = {
        "original": {
            "hr": original_hr,
            "ndcg": original_ndcg,
            "surrogate_hr": original_surrogate_hr,
            "surrogate_ndcg": original_surrogate_ndcg,
            "attack_success": original_attack_success,
            "attack_metrics": original_attack_metrics,
        },
        "detection": {
            "threshold": hoda.threshold,
            "attack_distance": distance,
            "benign_distance": benign_distance,
            "attack_detected": is_attack,
            "false_positive": is_benign_attack,
            "num_queries": args.detection_num_seqs,
        },
    }

    # Save metrics
    np.save(os.path.join(args.output_dir, "hoda_metrics.npy"), metrics)
    print(f"Results saved to {args.output_dir}")


def main():
    args = parse_args()
    apply_hoda_defense_and_evaluate(args)


if __name__ == "__main__":
    main()
