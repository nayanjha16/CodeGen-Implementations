import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluate import compute_exact_match

def main() -> None:
    """
    Main entry point for the evaluation script.
    
    Note:
        Currently a placeholder. Implement evaluation loop as needed.
    """
    print("Running Evaluation...")
    # Placeholder for evaluation loop
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
