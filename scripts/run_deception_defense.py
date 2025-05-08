#!/usr/bin/env python
"""
Wrapper script to run the HoneypotNet defense with proper path setup.

This script ensures that the Python path includes the project root directory
before running the HoneypotNet defense, fixing the ModuleNotFoundError.
"""

import os
import sys
import subprocess


def main():
    """Set up the environment and run the HoneypotNet defense script."""
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Print debugging information
    print(f"Current script directory: {current_dir}")
    print(f"Project root directory: {project_root}")

    # Ensure the project root is in the Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to Python path")

    # Check if required modules are importable
    try:
        # Try importing the problematic module
        from src.models.base_model import SimpleSequentialRecommender
        from src.data.data_utils import load_movielens, create_train_val_test_splits
        from src.attack.model_extraction import ModelExtractionAttack
        from src.defense.honeypotnet_defense import HoneypotNetDefense

        print("✅ Successfully imported required modules")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTrying to fix the path...")

        # Create __init__.py files if they don't exist to make Python treat directories as packages
        for path in ["src", "src/models", "src/data", "src/attack", "src/defense"]:
            init_file = os.path.join(project_root, path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write("# Auto-generated __init__.py file\n")
                print(f"Created {init_file}")

        # Try importing again
        try:
            from src.models.base_model import SimpleSequentialRecommender
            from src.data.data_utils import load_movielens, create_train_val_test_splits
            from src.attack.model_extraction import ModelExtractionAttack
            from src.defense.honeypotnet_defense import HoneypotNetDefense

            print("✅ Successfully imported required modules after fix")
        except ImportError as e:
            print(f"❌ Import error still persists: {e}")
            print(
                "\nPlease run the following command to add the project directory to PYTHONPATH:"
            )
            print(f"export PYTHONPATH={project_root}:$PYTHONPATH")
            return 1

    # Now fix the script file to use the correct function name
    honeypotnet_script = os.path.join(current_dir, "run_honeypotnet_defense.py")

    # Backup the original file
    backup_script = honeypotnet_script + ".bak"
    if not os.path.exists(backup_script):
        with open(honeypotnet_script, "r") as src, open(backup_script, "w") as backup:
            backup.write(src.read())
        print(f"Created backup of original script at {backup_script}")

    # Fix the import in the script
    try:
        with open(honeypotnet_script, "r") as f:
            content = f.read()

        # Fix any typo in the function name (extra 's')
        fixed_content = content.replace(
            "create_train_val_test_splitss", "create_train_val_test_splits"
        )

        # Replace the incorrect import with the correct one
        fixed_content = fixed_content.replace(
            "from src.data.data_utils import load_movielens, create_train_val_test_split",
            "from src.data.data_utils import load_movielens, create_train_val_test_splits",
        )

        # Update any function calls if needed
        fixed_content = fixed_content.replace(
            "create_train_val_test_split(", "create_train_val_test_splits("
        )

        # Fix the function call to ensure it uses user_sequences properly
        fixed_content = fixed_content.replace(
            "train_data, val_data, test_data = create_train_val_test_splits(data)",
            'train_data, val_data, test_data = create_train_val_test_splits(data["user_sequences"])',
        )

        with open(honeypotnet_script, "w") as f:
            f.write(fixed_content)

        print(
            "✅ Fixed import statement and function calls in HoneypotNet defense script"
        )
    except Exception as e:
        print(f"❌ Error fixing script: {e}")
        return 1

    # Get command line arguments
    args = sys.argv[1:]

    print(f"\nRunning HoneypotNet defense script: {honeypotnet_script}")
    print(f"With arguments: {' '.join(args)}")

    # Use the current Python interpreter to run the script
    cmd = [sys.executable, honeypotnet_script] + args

    # Execute the script and return its exit code
    return subprocess.call(cmd, env=dict(os.environ, PYTHONPATH=project_root))


if __name__ == "__main__":
    sys.exit(main())
