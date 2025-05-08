import torch
import pickle
import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pad_sequence(sequence, max_len=10, pad_value=0):
    """Pad sequence to max_len"""
    sequence = sequence[-max_len:] if len(sequence) > max_len else sequence
    padding = [pad_value] * (max_len - len(sequence))
    return sequence + padding


def main():
    # Create directory
    os.makedirs("attack_results", exist_ok=True)

    # Load user sequences
    with open("data/user_sequences.pkl", "rb") as f:
        user_sequences = pickle.load(f)

    # Take a subset for attack queries
    random.seed(42)  # For reproducibility
    attack_sequences = random.sample(user_sequences, min(1000, len(user_sequences)))

    # Convert to tensors with padding
    attack_queries = []
    for seq in attack_sequences:
        if len(seq) >= 5:  # Only use sequences of reasonable length
            # Create a slightly perturbed version to simulate attack
            perturbed = seq.copy()
            if len(perturbed) > 5:
                # Swap two items to simulate a small perturbation
                idx1, idx2 = random.sample(range(len(perturbed)), 2)
                perturbed[idx1], perturbed[idx2] = perturbed[idx2], perturbed[idx1]

            # Pad to standard length
            padded_seq = pad_sequence(perturbed, max_len=10)
            attack_queries.append(torch.tensor(padded_seq, dtype=torch.long))

    # Stack and save
    attack_queries = torch.stack(attack_queries)
    torch.save(attack_queries, "attack_results/jba_queries.pt")
    print(f"Created attack queries dataset with {len(attack_queries)} sequences")


if __name__ == "__main__":
    main()
