#!/usr/bin/env python
"""
Generate all figures for the deception-based defense (HoneypotNet) implementation.

This script runs all the individual figure generators to create a complete set of
visualizations for the HoneypotNet defense against model extraction attacks.
"""

import os
import subprocess
import sys


def main():
    """Run all figure generation scripts for the deception-based defense."""
    print("Generating figures for deception-based defense (HoneypotNet)...")

    # Ensure the output directory exists
    os.makedirs("figures/deception", exist_ok=True)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # List of scripts to run
    scripts = [
        "generate_honeypotnet_architecture.py",
        "generate_utility_backdoor_tradeoff.py",
        "generate_attack_success_comparison.py",
    ]

    # Run each script
    for script in scripts:
        script_path = os.path.join(script_dir, script)
        print(f"\nRunning {script}...")
        try:
            # Run the script as a subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                text=True,
                capture_output=True,
            )
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")

    print("\nAll deception-based defense figures generated successfully.")
    print("\nFigures saved in figures/deception/:")
    for filename in os.listdir("figures/deception"):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
