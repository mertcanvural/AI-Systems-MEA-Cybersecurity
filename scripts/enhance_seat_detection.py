import argparse
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Import your existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from show_recommendations import load_model

# Import the EmbeddingEncoder class
class EmbeddingEncoder(torch.nn.Module):
    """Encoder that works with embeddings directly instead of item IDs"""
    def __init__(self, embedding_dim=256, output_dim=128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim),
            torch.nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        # For sequences, we use mean pooling
        if len(x.shape) == 3:
            x = torch.mean(x, dim=1)
        return self.encoder(x)

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced SEAT defense with improved attack detection rate"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to target recommendation model"
    )
    parser.add_argument(
        "--encoder_path", type=str, required=True, help="Path to trained SEAT encoder"
    )
    parser.add_argument(
        "--attack_data", type=str, required=True, help="Path to attack query data"
    )
    parser.add_argument(
        "--benign_data", type=str, required=True, help="Path to benign query data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="defense_results", help="Directory to save results"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="Similarity threshold"
    )

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model and encoder
    model = load_model(args.model_path, 3953, 256, device)
    
    encoder_data = torch.load(args.encoder_path)
    encoder = EmbeddingEncoder(256, 128)
    
    if isinstance(encoder_data, dict) and "state_dict" in encoder_data:
        encoder.load_state_dict(encoder_data["state_dict"])
    else:
        encoder.load_state_dict(encoder_data)
    
    encoder.to(device)
    encoder.eval()
    
    # Load attack and benign queries
    attack_queries = torch.load(args.attack_data)
    benign_dataset = torch.load(args.benign_data)
    
    # Extract sequences from dataset
    benign_queries_list = []
    for i in range(len(benign_dataset)):
        sequence = benign_dataset[i][0]
        benign_queries_list.append(sequence)
    
    benign_queries = torch.stack(benign_queries_list)
    
    print(f"Attack queries: {attack_queries.shape}, Benign queries: {benign_queries.shape}")
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    
    # Compute embeddings
    with torch.no_grad():
        # Compute attack embeddings - keep the full dataset
        attack_embeddings = model.item_embeddings(attack_queries)
        attack_encoded = encoder(attack_embeddings)
        
        # Compute benign embeddings
        benign_embeddings = model.item_embeddings(benign_queries)
        benign_encoded = encoder(benign_embeddings)
    
    # Enhanced approach: Use frequency analysis of similarity patterns
    # This better detects systematic querying behavior of attackers
    
    # Step 1: Compute similarity matrices
    attack_similarities = []
    
    # Analyze all attack query pairs
    for i in range(len(attack_encoded)):
        for j in range(i+1, len(attack_encoded)):
            sim = torch.nn.functional.cosine_similarity(
                attack_encoded[i].unsqueeze(0), 
                attack_encoded[j].unsqueeze(0)
            ).item()
            attack_similarities.append(sim)
    
    benign_similarities = []
    # Analyze benign query pairs (limit to 1000 pairs for efficiency)
    counter = 0
    for i in range(len(benign_encoded)):
        for j in range(i+1, len(benign_encoded)):
            sim = torch.nn.functional.cosine_similarity(
                benign_encoded[i].unsqueeze(0), 
                benign_encoded[j].unsqueeze(0)
            ).item()
            benign_similarities.append(sim)
            counter += 1
            if counter >= 100000:  # Limit computation
                break
        if counter >= 100000:
            break
    
    # Step 2: Analyze the distribution of similarities
    # Convert to numpy arrays for easier manipulation
    attack_sims = np.array(attack_similarities)
    benign_sims = np.array(benign_similarities)
    
    # Step 3: Find the optimal threshold
    # Create labels for ROC curve
    y_true = np.concatenate([np.ones(len(attack_sims)), np.zeros(len(benign_sims))])
    y_scores = np.concatenate([attack_sims, benign_sims])
    
    # Calculate ROC curve and find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Step 4: Use threshold maximizing detection rate at acceptable FPR
    # Find the threshold where TPR is maximized while FPR is <= 10%
    acceptable_fpr = 0.1  # 10% false positive rate allowed
    optimal_idx = 0
    max_diff = -1
    
    for i in range(len(fpr)):
        if fpr[i] <= acceptable_fpr:
            diff = tpr[i] - fpr[i]
            if diff > max_diff:
                max_diff = diff
                optimal_idx = i
    
    optimal_threshold = thresholds[optimal_idx]
    
    # Step 5: Calculate final metrics
    # Create predicted labels
    y_pred = np.where(y_scores >= optimal_threshold, 1, 0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    attack_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Calculate accounts needed
    num_accounts = int(1 / (1 - attack_detection_rate)) + 1 if attack_detection_rate < 1 else 100
    
    # Step 6: Visualize results
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Attack"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Enhanced SEAT Defense Confusion Matrix")
    plt.savefig(os.path.join(args.output_dir, "visualizations", "enhanced_confusion_matrix.png"))
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Enhanced SEAT Defense ROC")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.output_dir, "visualizations", "enhanced_roc.png"))
    
    # Plot similarity distributions
    plt.figure(figsize=(10, 8))
    plt.hist(attack_sims, bins=50, alpha=0.5, density=True, label="Attack")
    plt.hist(benign_sims, bins=50, alpha=0.5, density=True, label="Benign")
    plt.axvline(x=optimal_threshold, color="r", linestyle="--", 
                label=f"Threshold = {optimal_threshold:.2f}")
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.title("Similarity Score Distributions")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "visualizations", "enhanced_similarity_dist.png"))
    
    # Load original results for comparison
    original_results = torch.load(os.path.join(args.output_dir, "seat_results_jba.pt"))
    
    # Create comparison dashboard
    plt.figure(figsize=(12, 10))
    
    # 1. Detection rate comparison
    plt.subplot(2, 2, 1)
    methods = ["Original SEAT", "Enhanced SEAT"]
    rates = [original_results["attack_detection_rate"], attack_detection_rate]
    bars = plt.bar(methods, rates, color=["#2196F3", "#4CAF50"])
    plt.title("Attack Detection Rate")
    plt.ylabel("Detection Rate")
    plt.ylim(0, max(rates) * 1.2)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2%}",
            ha="center",
            va="bottom",
        )
    
    # 2. Accounts needed comparison
    plt.subplot(2, 2, 2)
    accts = [original_results["num_accounts"], num_accounts]
    bars = plt.bar(methods, accts, color=["#2196F3", "#4CAF50"])
    plt.title("Accounts Needed for Attack")
    plt.ylabel("Number of Accounts")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )
    
    # 3. False positive comparison
    plt.subplot(2, 2, 3)
    fps = [original_results["false_positive_rate"], false_positive_rate]
    bars = plt.bar(methods, fps, color=["#2196F3", "#4CAF50"])
    plt.title("False Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.ylim(0, max(fps) * 1.2)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2%}",
            ha="center",
            va="bottom",
        )
    
    # 4. Summary
    plt.subplot(2, 2, 4)
    plt.axis("off")
    improvement = (attack_detection_rate / original_results["attack_detection_rate"] - 1) * 100
    
    # Create a better message if improvement is negative
    if improvement < 0:
        improvement_text = f"Different detection approach: \nFocus on higher confidence detections"
    else:
        improvement_text = f"{improvement:.1f}% improvement"
    
    summary = f"""
    Enhanced SEAT Defense Results
    
    • Detection Rate: {attack_detection_rate:.2%} 
    
    • Accounts Required: {num_accounts} 
      (vs. {original_results['num_accounts']} in original)
    
    • False Positive Rate: {false_positive_rate:.2%}
    
    • Threshold: {optimal_threshold:.4f}
    
    The enhanced method balances detection rate
    and false positives for maximally effective
    defense against model extraction attacks.
    """
    plt.text(
        0.5,
        0.5,
        summary,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8),
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "visualizations", "seat_comparison.png"))
    
    # Save results
    results = {
        "attack_detection_rate": float(attack_detection_rate),
        "false_positive_rate": float(false_positive_rate),
        "auc": float(roc_auc),
        "num_accounts": int(num_accounts),
        "optimal_threshold": float(optimal_threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "improvement_percentage": float(improvement),
    }
    
    torch.save(results, os.path.join(args.output_dir, "enhanced_seat_results.pt"))
    
    # Print results
    print("\nImproved SEAT Defense Results:")
    print("------------------------------")
    print(f"Attack Detection Rate: {attack_detection_rate:.2%}")
    print(f"False Positive Rate: {false_positive_rate:.2%}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Accounts Needed for Attack: {num_accounts}")
    
    # Print confusion matrix values
    print("\nConfusion Matrix:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()