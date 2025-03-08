# Prevention-Based-Cybersecurity-GRO-Defense-for-AI-Systems

Implementation of the paper "Defense Against Model Extraction Attacks on Recommender Systems"

## Project Structure

```
gro_defense/
├── data/                 # Data storage
├── src/
│   ├── data/            # Data handling utilities
│   ├── models/          # Model implementations
│   └── utils/           # Helper functions
├── main.py              # Main script
└── download_datasets.py # Script to download datasets
```

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the datasets:

```bash
python download_datasets.py --datasets ml-1m
```

4. Run the main script:

```bash
python main.py
```

## Current Status

This is the initial implementation with data utilities. Model implementations coming soon.

## Requirements

```
numpy==1.24.3
pandas==2.0.2
torch==2.0.1
tqdm==4.65.0
requests==2.31.0
```
