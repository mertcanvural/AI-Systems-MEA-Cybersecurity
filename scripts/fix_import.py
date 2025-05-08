#!/usr/bin/env python
"""
Quick script to fix the import statement in run_honeypotnet_defense.py
"""

import os
import sys


def main():
    """Fix the import statement in run_honeypotnet_defense.py"""
    # Path to the script file
    script_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_honeypotnet_defense.py"
    )

    # Read the content of the file
    with open(script_file, "r") as f:
        content = f.read()

    # Fix the import statement
    fixed_content = content.replace(
        "from src.data.data_utils import load_movielens, create_train_val_test_splitss",
        "from src.data.data_utils import load_movielens, create_train_val_test_splits",
    )

    # Write the fixed content back to the file
    with open(script_file, "w") as f:
        f.write(fixed_content)

    print(f"✅ Fixed import statement in {script_file}")


if __name__ == "__main__":
    main()
