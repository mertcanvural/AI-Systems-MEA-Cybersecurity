#!/usr/bin/env python
"""
A simple script to check that imports work correctly
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the function to confirm it exists
try:
    from src.data.data_utils import create_train_val_test_splits

    print("✅ Successfully imported create_train_val_test_splits")
except ImportError as e:
    print(f"❌ Import error: {e}")


def main():
    print("Import check completed")


if __name__ == "__main__":
    main()
