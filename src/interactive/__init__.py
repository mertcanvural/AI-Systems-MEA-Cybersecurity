"""
Scripts for running the recommendation system.

The scripts in this package provide functionality for:
- Downloading the MovieLens dataset (download_datasets.py)
- Analyzing the dataset (analyze_dataset.py)
- Training a recommendation model (train_improved_model.py)
- Evaluating model performance (evaluate_model.py)
- Visualizing training metrics (visualize_performance.py)
- Interactive recommendation (interactive_model_recommend.py)
"""

# Export module functions for easier imports
from ..utils.train_improved_model import main as train_main
from ..models.evaluate_model import main as evaluate_main
from ..utils.visualize_performance import main as visualize_main
from .interactive_model_recommend import main as interactive_main
from ..data.analyze_dataset import main as analyze_main

# Import download function directly
try:
    from ..data.download_datasets import download_movielens
except ImportError:
    pass
