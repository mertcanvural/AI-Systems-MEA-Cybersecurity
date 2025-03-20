import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model performance metrics")
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="figures/training_metrics.npz",
        help="Path to saved training metrics file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save output figures",
    )
    return parser.parse_args()


def main(args=None):
    # Parse arguments
    if args is None:
        args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if metrics file exists
    if not os.path.exists(args.metrics_file):
        print(f"Metrics file not found: {args.metrics_file}")
        print("Please train the model first using train_improved_model.py")
        return None

    # Load metrics
    metrics = np.load(args.metrics_file)
    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    train_hit_rates = metrics["train_hit_rates"]
    val_hit_rates = metrics["val_hit_rates"]

    # Generate epochs array if it doesn't exist
    if "epochs" not in metrics:
        print("'epochs' key not found in metrics file, generating it...")
        # 1-indexed epochs
        epochs = np.arange(1, len(train_losses) + 1)
    else:
        epochs = metrics["epochs"]

    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )

    # 1. Training Progress Plot - Combined Metrics
    plt.figure(figsize=(15, 6))

    # Plot loss
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, train_losses, "b-", linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot hit rate
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, train_hit_rates, "b-", linewidth=2, label="Train Hit@10")
    ax2.plot(epochs, val_hit_rates, "r-", linewidth=2, label="Validation Hit@10")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Hit@10")
    ax2.set_title("Training and Validation Hit@10")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_progress.png"), dpi=300)
    plt.close()

    # 2. Learning Curves - Log Scale
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", linewidth=2, label="Train Loss")
    plt.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training and Validation Loss (Log Scale)")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss_log_scale.png"), dpi=300)
    plt.close()

    # 3. Generalization Gap Plot
    plt.figure(figsize=(15, 6))
    loss_gap = np.abs(np.array(val_losses) - np.array(train_losses))
    hit_gap = np.abs(np.array(val_hit_rates) - np.array(train_hit_rates))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_gap, "g-", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Gap")
    plt.title("Generalization Gap (Loss)")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hit_gap, "g-", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Hit@10 Gap")
    plt.title("Generalization Gap (Hit@10)")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "generalization_gap.png"), dpi=300)
    plt.close()

    # 4. Combined Metrics with Annotations
    plt.figure(figsize=(12, 8))

    # Find the best epoch based on validation loss
    best_epoch_idx = np.argmin(val_losses)
    best_epoch = epochs[best_epoch_idx]
    best_val_loss = val_losses[best_epoch_idx]

    # Main plot with losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, "b-", linewidth=2, label="Train Loss")
    plt.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")
    plt.axvline(
        x=best_epoch,
        color="g",
        linestyle="--",
        alpha=0.7,
        label=f"Best Epoch: {best_epoch}",
    )
    plt.scatter(best_epoch, best_val_loss, color="g", s=100, zorder=5)
    plt.annotate(
        f"Best Val Loss: {best_val_loss:.4f}",
        xy=(best_epoch, best_val_loss),
        xytext=(best_epoch + 0.5, best_val_loss + 0.2),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
        fontsize=10,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss with Best Model Highlighted")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Hit rate plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_hit_rates, "b-", linewidth=2, label="Train Hit@10")
    plt.plot(epochs, val_hit_rates, "r-", linewidth=2, label="Validation Hit@10")
    plt.axvline(
        x=best_epoch,
        color="g",
        linestyle="--",
        alpha=0.7,
        label=f"Best Epoch: {best_epoch}",
    )
    plt.scatter(best_epoch, val_hit_rates[best_epoch_idx], color="g", s=100, zorder=5)
    plt.annotate(
        f"Val Hit@10: {val_hit_rates[best_epoch_idx]:.4f}",
        xy=(best_epoch, val_hit_rates[best_epoch_idx]),
        xytext=(best_epoch + 0.5, val_hit_rates[best_epoch_idx] + 0.005),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
        fontsize=10,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Hit@10")
    plt.title("Training and Validation Hit@10 with Best Model Highlighted")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "best_model_metrics.png"), dpi=300)
    plt.close()

    print(f"Visualizations saved in {args.output_dir}/")

    # Print key statistics
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = np.min(val_losses)
    best_hit_rate = val_hit_rates[best_epoch - 1]

    print("Best model performance:")
    print(f"  Epoch: {best_epoch}")
    print(f"  Validation Loss: {best_val_loss:.4f}")
    print(f"  Validation Hit@10: {best_hit_rate:.4f}")
    print(f"  Final Train Loss: {train_losses[-1]:.4f}")
    print(f"  Final Train Hit@10: {train_hit_rates[-1]:.4f}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_hit_rate": best_hit_rate,
        "final_train_loss": train_losses[-1],
        "final_train_hit_rate": train_hit_rates[-1],
    }


if __name__ == "__main__":
    main()
