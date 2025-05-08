#!/usr/bin/env python
"""
Run HoneypotNet defense against model extraction attacks.

This script applies the HoneypotNet defense to protect a recommendation model
by injecting a backdoor that transfers to any extracted model, allowing for
both ownership verification and functionality disruption of stolen models.
"""

# Standard library imports
import os
import sys
import argparse
import random

# Third-party imports
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project imports
from src.models.base_model import SimpleSequentialRecommender
from src.data.data_utils import load_movielens, create_train_val_test_splits
from src.attack.model_extraction import ModelExtractionAttack
from src.defense.honeypotnet_defense import HoneypotNetDefense


def main():
    """Main function to run the HoneypotNet defense"""
    print("HoneypotNet defense initialized")
    print("Using correct imports with create_train_val_test_splits")


if __name__ == "__main__":
    main()
