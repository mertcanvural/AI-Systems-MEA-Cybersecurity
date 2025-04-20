# Movie Recommendation System with Model Extraction Defense

This repository contains an implementation of a sequential movie recommendation system trained on the MovieLens-1M dataset, along with defense mechanisms against model extraction attacks.

## Overview

The recommendation system predicts which movies a user might enjoy based on their viewing history. We've also implemented and evaluated defenses against model extraction attacks, where attackers try to steal the functionality of recommendation models through API queries.

## Key Components

- **Recommendation Model**: A neural network that predicts the next movie a user might want to watch
- **Model Extraction Attack**: Implementation of methods to steal model behavior through black-box queries
- **GRO Defense**: Gradient-based Ranking Optimization defense that prevents successful model extraction
- **SEAT Defense**: Similarity Encoder by Adversarial Training for detecting extraction attack attempts

## Defense Implementation

### GRO Prevention Defense

The Gradient-based Ranking Optimization (GRO) defense works by:

1. Using a student model to simulate potential attackers
2. Converting ranking lists to differentiable swap matrices
3. Computing gradients to maximize the attacker's (student) model loss
4. Training the target model to both perform well and resist extraction attempts

Core defense code:

```python
# Core of the gradient-based ranking defense
def compute_swap_loss(target_logits, student_logits, margin_swap=0.3):
    swap_matrix_target = create_swap_matrix(target_logits)
    swap_matrix_student = create_swap_matrix(student_logits)

    swap_loss = torch.mean(torch.relu(margin_swap -
                         torch.abs(swap_matrix_target - swap_matrix_student)))

    return swap_loss

loss = rec_loss + lambda_swap * swap_loss
```

### SEAT Detection Defense

The SEAT detection defense works by:

1. Training a similarity encoder to recognize suspicious query patterns
2. Monitoring user accounts for suspiciously similar query pairs
3. Flagging accounts that exceed a threshold of similar pairs
4. Tracking the number of accounts needed for a successful attack

The defense is non-invasive (doesn't modify recommendations) and has no negative impact on legitimate users.

## Defense Results

### GRO Prevention Results

- **Utility Preservation**: 96.91% (defended model maintains most utility)
- **Attack Success Reduction**: Original attack success 98.8% → Defended 55.7%
- **Defense Effectiveness**: 43.57% reduction in attack success
- **Recommendation Overlap**: Original attack overlap@10: 0.90 → Defended: 0.25

### SEAT Detection Results

- **Attack Detection Rate**: 100% (for single-account attacks)
- **False Positive Rate**: 0.0000% (no benign queries flagged as malicious)
- **Accounts Needed**: 10 (minimum number of accounts an attacker would need to distribute queries to evade detection)

## Usage

### Training with GRO Defense

```bash
python src/defense/apply_defense.py --target-model checkpoints/best_model.pt --lambda-swap 5.0
```

### Running SEAT Detection

```bash
python scripts/run_seat_defense.py \
  --model_path checkpoints/best_model.pt \
  --encoder_path defense_results/seat_encoder.pt \
  --attack_data defense_data/seat_attack_data.pt \
  --benign_data defense_data/seat_test_data.pt
```

### Model Extraction Attack

```bash
python src/attack/model_extraction.py --target-model checkpoints/best_model.pt --queries 500
```

### Comparing Recommendations

```bash
python show_recommendations.py
```

## Requirements

```bash
pip install torch numpy pandas matplotlib seaborn tqdm networkx
```

## Project Structure

```
.
├── data/                       # Dataset storage
├── src/
│   ├── models/                 # Recommendation models
│   ├── attack/                 # Model extraction attack
│   │   └── model_extraction.py # Attack implementation
│   └── defense/                # Defense mechanisms
│       ├── SEAT/               # SEAT detection mechanism
│       └── gro_defense.py      # GRO defense implementation
├── checkpoints/                # Model checkpoints
├── defense_results/            # Defense evaluation results
├── attack_results/             # Attack evaluation results
└── figures/                    # Performance visualizations
```

For more detailed information about the basic recommendation system, see the original documentation below.

## Model Architecture

The core of the system is a `SimpleSequentialRecommender` model with the following components:

- Movie embedding layer (256-dimensional)
- Dropout layer (0.1 dropout rate)
- Fully connected output layer

The model works by first converting each movie ID into a dense vector representation (embedding), then averaging these embeddings to create a representation of the user's taste, and finally using a fully connected layer to transform this representation into prediction scores for all possible movies.

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
