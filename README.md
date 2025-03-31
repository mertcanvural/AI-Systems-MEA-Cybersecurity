# Movie Recommendation System

This repository contains an implementation of a model-based movie recommendation system trained on the MovieLens-1M dataset. The system uses a neural network to predict which movies a user might enjoy based on their viewing history.

## Overview

The recommendation system employs a sequential neural network model that takes a sequence of previously watched movies as input and predicts the next movie the user might want to watch. The model is based on embedding representations of movies.

## Model Architecture

The core of the system is a `SimpleSequentialRecommender` model with the following components:

- Movie embedding layer (256-dimensional)
- Dropout layer (0.1 dropout rate)
- Fully connected output layer

The model works by first converting each movie ID into a dense vector representation (embedding), then averaging these embeddings to create a representation of the user's taste, and finally using a fully connected layer to transform this representation into prediction scores for all possible movies.

## Requirements

Before running the system, make sure you have the required dependencies installed:

```bash
pip install torch numpy pandas matplotlib seaborn tqdm tabulate requests
```

## Usage

You can interact with the system using the `main.py` script which provides several commands:

### Downloading the Dataset

To download the MovieLens dataset:

```bash
python main.py download --datasets ml-1m
```

Optional parameters:

- `--output-dir`: Output directory for datasets (default: data)
- `--datasets`: Datasets to download (choices: ml-1m, ml-20m)

### Analyzing the Dataset

To analyze the MovieLens dataset and generate visualizations:

```bash
python main.py analyze --dataset ml-1m
```

Optional parameters:

- `--data-dir`: Directory containing the datasets (default: data)
- `--dataset`: Dataset to analyze (default: ml-1m)
- `--output-dir`: Directory to save figures (default: figures)
- `--min-seq-length`: Minimum sequence length to keep (default: 5)

### Training a Model

To train a new recommendation model:

```bash
python main.py train --epochs 30 --batch-size 128 --embedding-dim 256
```

Optional parameters:

- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay for regularization (default: 1e-5)
- `--embedding-dim`: Embedding dimension (default: 256)
- `--device`: Device to use for training (default: cpu)

### Evaluating a Model

To evaluate a trained model:

```bash
python main.py evaluate --model-path checkpoints/best_model.pt
```

Optional parameters:

- `--model-path`: Path to the trained model (default: checkpoints/best_model.pt)
- `--embedding-dim`: Embedding dimension (default: 256)
- `--top-k`: Number of recommendations to generate (default: 10)

### Visualizing Performance

To visualize the model's training performance:

```bash
python main.py visualize
```

Optional parameters:

- `--metrics-file`: Path to the metrics file (default: figures/training_metrics.npz)
- `--output-dir`: Directory to save the visualizations (default: figures)

### Interactive Recommendation

To use the interactive recommendation system:

```bash
python main.py interactive --model checkpoints/best_model.pt
```

Optional parameters:

- `--model`: Path to the trained model (default: checkpoints/best_model.pt)
- `--embedding-dim`: Embedding dimension (default: 256)
- `--top-k`: Number of recommendations to show (default: 10)

#### Interactive Commands

When using the interactive recommendation system, you have the following commands:

- `search <query>`: Search for movies matching a query
- `add <id>`: Add a movie to your sequence
- `remove <id or index>`: Remove a movie from your sequence
- `clear`: Clear your sequence
- `recommend`: Get recommendations based on your sequence
- `show`: Show your current sequence
- `popular`: Show popular movies
- `exit`: Exit the program

## Complete Workflow Example

```bash
# Download the dataset
python main.py download --datasets ml-1m

# Analyze the dataset
python main.py analyze --dataset ml-1m

# Train a model
python main.py train --epochs 30 --embedding-dim 256

# Evaluate the model
python main.py evaluate

# Generate visualizations
python main.py visualize

# Run the interactive recommendation system
python main.py interactive
```

## Project Structure

The project is organized as follows:

```
.
├── data/                  # Dataset storage
│   └── ml-1m/             # MovieLens 1M dataset
├── src/                   # Source code
│   ├── data/              # Data processing utilities
│   ├── models/            # Model implementations
│   │   └── base_model.py  # SimpleSequentialRecommender definition
│   ├── scripts/           # Runnable scripts
│   └── utils/             # Helper utilities
├── checkpoints/           # Saved model checkpoints
├── figures/               # Output visualizations
└── main.py                # Main entry point
```

## Performance Metrics

The model achieves:

- Test Hit@10: ~4.8% (chance of the correct next movie appearing in the top 10 recommendations)
- Training time: ~5 minutes on CPU
- Genre consistency: ~0.28 average Jaccard similarity between input and recommendation genres

## Key Observations

1. The model tends to recommend popular movies across various input sequences
2. There's a bias toward recent movies in the dataset (1999-2000 releases)
3. Genre consistency is moderate, with action/sci-fi sequences receiving the most relevant recommendations
4. The model struggles with niche genres like children's movies, often recommending adult content instead

## Dataset Information

This system uses the MovieLens-1M dataset which includes:

- 1 million ratings from 6000+ users on 4000 movies
- Rating data format: UserID::MovieID::Rating::Timestamp
- Movie data format: MovieID::Title::Genres
