#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import torch


def setup_environment():
    """Setup environment variables and ensure directories exist"""
    # Ensure data and checkpoint directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Add src to path if not already there
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if src_path not in sys.path:
        sys.path.append(src_path)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Movie Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py download --datasets ml-1m
  python main.py analyze --data-dir data --dataset ml-1m
  python main.py train --epochs 30 --batch-size 128 --embedding-dim 256
  python main.py evaluate --model-path checkpoints/best_model.pt
  python main.py visualize
  python main.py interactive --model checkpoints/best_model.pt
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download dataset parser
    download_parser = subparsers.add_parser(
        "download", help="Download MovieLens dataset"
    )
    download_parser.add_argument(
        "--output-dir", type=str, default="data", help="Output directory for datasets"
    )
    download_parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["ml-1m"],
        choices=["ml-1m", "ml-20m"],
        help="Datasets to download",
    )

    # Analyze dataset parser
    analyze_parser = subparsers.add_parser("analyze", help="Analyze MovieLens dataset")
    analyze_parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory containing the datasets"
    )
    analyze_parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
        choices=["ml-1m", "ml-20m"],
        help="Dataset to analyze",
    )
    analyze_parser.add_argument(
        "--output-dir", type=str, default="figures", help="Directory to save figures"
    )
    analyze_parser.add_argument(
        "--min-seq-length", type=int, default=5, help="Minimum sequence length to keep"
    )

    # Train parser
    train_parser = subparsers.add_parser("train", help="Train the recommendation model")
    train_parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay"
    )
    train_parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension"
    )
    train_parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    train_parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    train_parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model",
    )
    train_parser.add_argument(
        "--disable-early-stopping", action="store_true", help="Disable early stopping"
    )

    # Evaluate parser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model",
    )
    eval_parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension"
    )
    eval_parser.add_argument(
        "--top-k", type=int, default=10, help="Number of recommendations to show"
    )
    eval_parser.add_argument(
        "--use-genre-bias",
        action="store_true",
        help="Use genre bias for recommendations",
    )

    # Visualize parser
    vis_parser = subparsers.add_parser("visualize", help="Visualize model performance")
    vis_parser.add_argument(
        "--metrics-file",
        type=str,
        default="figures/training_metrics.npz",
        help="Path to metrics file",
    )
    vis_parser.add_argument(
        "--output-dir", type=str, default="figures", help="Output directory for figures"
    )

    # Interactive parser
    interactive_parser = subparsers.add_parser(
        "interactive", help="Interactive recommendation"
    )
    interactive_parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    interactive_parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension"
    )
    interactive_parser.add_argument(
        "--top-k", type=int, default=10, help="Number of recommendations to show"
    )
    interactive_parser.add_argument(
        "--use-genre-bias",
        action="store_true",
        help="Use genre bias for recommendations",
    )

    # Compare genre-based recommendations
    genre_parser = subparsers.add_parser(
        "genre-compare", help="Compare standard vs genre-biased recommendations"
    )
    genre_parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    genre_parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension"
    )
    genre_parser.add_argument(
        "--top-k", type=int, default=10, help="Number of recommendations to show"
    )

    # Performance visualization command
    perf_parser = subparsers.add_parser(
        "performance", help="Visualize model performance metrics"
    )
    perf_parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    perf_parser.add_argument(
        "--metrics-path",
        type=str,
        default="checkpoints/metrics.json",
        help="Path to training metrics JSON file",
    )
    perf_parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save output figures",
    )
    perf_parser.add_argument(
        "--embedding-dim", type=int, default=256, help="Embedding dimension"
    )

    return parser.parse_args()


def run_download(args):
    """Run the download script"""
    try:
        from src.data import download_datasets

        # Call the main function directly
        print(f"Downloading datasets: {', '.join(args.datasets)}...")
        return download_datasets.download_movielens(args.output_dir, args.datasets[0])
    except ImportError as e:
        print(f"Error importing download module: {e}")
        fallback_run_command("src/data/download_datasets.py", vars(args))


def run_analyze(args):
    """Run the analyze script"""
    try:
        from src.data import analyze_dataset

        # Convert args to the expected format
        analyze_args = argparse.Namespace(
            data_dir=args.data_dir,
            dataset=args.dataset,
            output_dir=args.output_dir,
            min_seq_length=args.min_seq_length,
        )

        print(f"Analyzing dataset {args.dataset}...")
        return analyze_dataset.main(analyze_args)
    except ImportError as e:
        print(f"Error importing analyze module: {e}")
        fallback_run_command("src/data/analyze_dataset.py", vars(args))


def run_train(args):
    """Run the training script"""
    try:
        from src.utils.train_improved_model import main as train_main
    except ImportError:
        print("Could not import training module, trying fallback method")
        try:
            from src.train_model import main as train_main
        except ImportError:
            # Fallback to running the script directly
            for path in [
                "src/utils/train_improved_model.py",
                "src/train_model.py",
            ]:
                if os.path.exists(path):
                    fallback_run_command(path, vars(args))
                    return
            print("Could not find training script in any expected location")
            return

    print(
        f"Training model with {args.embedding_dim} dimensional embeddings for {args.epochs} epochs..."
    )
    train_args = argparse.Namespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        dropout=args.dropout,
        model_path=args.model_path,
        disable_early_stopping=getattr(args, "disable_early_stopping", False),
    )
    train_main(train_args)


def run_evaluate(args):
    """Run the evaluation script"""
    try:
        # First try the models module
        try:
            from src.models.evaluate_model import main as evaluate_main
        except ImportError:
            # Then try the interactive module
            from src.interactive.evaluate_model import main as evaluate_main

        # Convert args to the expected format
        eval_args = argparse.Namespace(
            model_path=args.model_path,
            embedding_dim=args.embedding_dim,
            top_k=args.top_k,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_genre_bias=getattr(args, "use_genre_bias", False),
        )

        print(f"Evaluating model from {args.model_path}...")
        return evaluate_main(eval_args)
    except ImportError as e:
        print(f"Error importing evaluate module: {e}")
        # Try different possible locations
        for path in [
            "src/models/evaluate_model.py",
            "src/interactive/evaluate_model.py",
        ]:
            if os.path.exists(path):
                fallback_run_command(path, vars(args))
                return
        print("Could not find evaluate_model.py in any expected location")


def run_visualize(args):
    """Run the visualization script"""
    try:
        from src.data.visualization import visualize_training_metrics

        print(f"Visualizing training metrics from {args.metrics_file}...")
        visualize_training_metrics(args.metrics_file, args.output_dir)
    except ImportError:
        # Fallback to running the script directly
        for path in ["src/data/visualization.py", "src/visualization.py"]:
            if os.path.exists(path):
                fallback_run_command(path, vars(args))
                return
        print("Could not find visualization script in any expected location")


def run_interactive(args):
    """Run the interactive recommendation script"""
    try:
        from src.interactive.interactive_model_recommend import main as interactive_main

        print("Starting interactive recommendation session...")
        return interactive_main(args)
    except ImportError:
        # Fallback to running the script directly
        for path in [
            "src/interactive/interactive_model_recommend.py",
            "src/interactive_model_recommend.py",
        ]:
            if os.path.exists(path):
                fallback_run_command(path, vars(args))
                return
        print(
            "Could not find interactive recommendation script in any expected location"
        )


def run_genre_compare(args):
    """Run the genre comparison script"""
    try:
        from src.evaluate_genre_bias import main as genre_main

        print(
            f"Comparing standard vs genre-biased recommendations with model from {args.model_path}..."
        )

        # Convert args to the expected format
        genre_args = argparse.Namespace(
            model_path=args.model_path,
            embedding_dim=args.embedding_dim,
            top_k=args.top_k,
        )

        return genre_main(genre_args)
    except ImportError as e:
        print(f"Error importing genre comparison module: {e}")
        # Try to find the script
        if os.path.exists("src/evaluate_genre_bias.py"):
            fallback_run_command("src/evaluate_genre_bias.py", vars(args))
            return
        print("Could not find evaluate_genre_bias.py")


def run_performance(args):
    """Run the model performance visualization script"""
    try:
        from src.models.visualize_model_performance import main as performance_main

        print(f"Visualizing model performance from {args.model_path}...")
        return performance_main(args)
    except ImportError as e:
        print(f"Error importing performance visualization module: {e}")
        # Try different possible locations
        for path in [
            "src/models/visualize_model_performance.py",
            "src/visualize_model_performance.py",
        ]:
            if os.path.exists(path):
                fallback_run_command(path, vars(args))
                return
        print("Could not find visualize_model_performance.py in any expected location")


def fallback_run_command(command, args_dict):
    """Fallback to running the script as a subprocess if import fails"""
    # If command is a full path, use it directly
    if command.endswith(".py"):
        script_path = command
    else:
        # Otherwise assume it's in src/scripts
        script_path = f"{command}.py"

    cmd = [sys.executable, script_path]
    for key, value in args_dict.items():
        if key.startswith("_") or key == "command":
            continue
        cmd.append(f"--{key.replace('_', '-')}")
        cmd.append(str(value))

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    """Main entry point"""
    setup_environment()
    args = parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "analyze":
        run_analyze(args)
    elif args.command == "visualize":
        run_visualize(args)
    elif args.command == "download":
        run_download(args)
    elif args.command == "genre-compare":
        run_genre_compare(args)
    elif args.command == "performance":
        run_performance(args)
    else:
        print("Please specify a command to run")
        print(
            "Available commands: train, evaluate, genre-compare, interactive, analyze, visualize, download, performance"
        )


if __name__ == "__main__":
    main()
