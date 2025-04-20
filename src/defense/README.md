# Defense Against Model Extraction Attacks

This directory contains implementations of defense methods for protecting recommender systems against model extraction attacks:

1. **GRO (Gradient-based Ranking Optimization)**: A defense that modifies model outputs to prevent attackers from creating accurate surrogate models.
2. **HODA (Hardness-Oriented Detection Approach)**: A detection method that identifies model extraction attacks by analyzing query patterns.

## GRO Defense

The GRO defense method works by:

1. Using a student model to simulate the attacker's behavior
2. Converting ranking lists to differentiable swap matrices
3. Computing gradients to maximize the loss of the student model
4. Training the target model to both perform well and fool potential attackers

### Implementation Details

- `gro_defense.py`: The core implementation of the GRO defense mechanism
- `apply_defense.py`: Script to apply the defense and evaluate its effectiveness

### Usage

To apply the GRO defense to a trained recommendation model:

```bash
python -m src.defense.apply_defense \
    --target-model checkpoints/best_model.pt \
    --data-path data/ml-1m/ratings.dat \
    --embedding-dim 256 \
    --num-epochs 5 \
    --batch-size 64 \
    --lambda-swap 5.0 \
    --attack-queries 1000 \
    --attack-epochs 10 \
    --output-dir defense_results
```

#### Parameters

- `--target-model`: Path to the target model checkpoint
- `--data-path`: Path to the dataset
- `--embedding-dim`: Embedding dimension for models
- `--num-epochs`: Number of epochs for defense training
- `--batch-size`: Batch size for training
- `--lambda-swap`: Weight of the swap loss (controls defense strength)
- `--attack-queries`: Number of queries for model extraction attack evaluation
- `--attack-epochs`: Number of epochs for training surrogate models
- `--output-dir`: Directory to save results

## HODA Defense

HODA is a detection-based defense method that analyzes the "hardness degree" of user queries to identify model extraction attacks. It's based on the paper "HODA: Hardness-Oriented Detection of Model Extraction Attacks".

### Key Concept

HODA works by:

1. Computing the "hardness degree" of queries using subclassifiers trained at different epochs
2. Creating histograms of hardness degrees for normal users and potential attackers
3. Using Pearson distance to detect abnormal query patterns indicative of extraction attacks

### Implementation Details

- `hoda_defense.py`: Core implementation of the HODA defense mechanism
- `apply_hoda_defense.py`: Script to apply and evaluate the defense
- `visualize_hoda_results.py`: Script to create visualizations of defense results

### Usage

To apply the HODA defense:

```bash
python -m src.defense.apply_hoda_defense \
    --target-model checkpoints/best_model.pt \
    --data-path data/ml-1m/ratings.dat \
    --embedding-dim 256 \
    --num-subclassifiers 5 \
    --train-subclassifiers \
    --training-epochs 100 \
    --batch-size 64 \
    --attack-queries 1000 \
    --attack-epochs 10 \
    --detection-num-seqs 100 \
    --output-dir defense_results
```

#### Parameters

- `--target-model`: Path to the target model checkpoint
- `--data-path`: Path to the dataset
- `--embedding-dim`: Embedding dimension for models
- `--num-subclassifiers`: Number of subclassifiers to use (5 or 11 recommended)
- `--train-subclassifiers`: Flag to train subclassifiers from scratch
- `--training-epochs`: Number of epochs for training subclassifiers
- `--batch-size`: Batch size for training
- `--attack-queries`: Number of queries for model extraction attack evaluation
- `--attack-epochs`: Number of epochs for training surrogate models
- `--detection-num-seqs`: Number of sequences to monitor for attack detection
- `--output-dir`: Directory to save results

### Visualization

To create visualizations of the HODA defense results:

```bash
python -m src.defense.visualize_hoda_results \
    --input-dir defense_results \
    --output-dir defense_results/visualizations
```

## Output and Interpretation

### GRO Defense Output

The GRO script generates:

1. `defended_model.pt`: The model protected with GRO defense
2. Various comparison metrics and visualizations

#### Key Metrics:

- **Utility Preservation**: How well the defended model maintains its recommendation quality
- **Defense Effectiveness**: How much the defense reduces the attacker's success rate
- **Rank Correlation**: Reduction in rank correlation between target and surrogate models
- **Overlap Metrics**: Reduction in recommendation overlap at different K values

### HODA Defense Output

The HODA script generates:

1. `subclassifiers/`: Directory containing trained subclassifiers
2. `hoda_defense.pt`: Saved state of the HODA defense
3. `hoda_metrics.npy`: Detailed metrics about detection performance
4. Visualization plots showing attack detection results

#### Key Metrics:

- **Detection Rate**: Percentage of attacks successfully detected
- **False Positive Rate**: Percentage of benign users incorrectly flagged as attackers
- **Pearson Distance**: Distance between normal and attack query histograms
- **Detection Threshold**: Value used to distinguish between normal and attack patterns

A successful HODA defense should:

- Detect a high percentage of attacks (ideally > 90%)
- Maintain a low false positive rate (ideally < 5%)
- Show clear separation between normal and attack query histograms
