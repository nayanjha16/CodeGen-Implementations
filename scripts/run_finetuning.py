"""
CLI script to run fine-tuning.
"""
import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train import train
import src.config as config

def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning for Text-to-SQL")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick test training (1 step)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum number of training steps (overrides epochs)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training examples")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("Running in DRY RUN mode (1 step only)...")
        config.TRAINING_ARGS["max_steps"] = 1

        config.TRAINING_ARGS["num_train_epochs"] = 1
        config.TRAINING_ARGS["logging_steps"] = 1
        config.TRAINING_ARGS["save_steps"] = 1
        
    if args.epochs:
        config.TRAINING_ARGS["num_train_epochs"] = args.epochs
        
    if args.batch_size:
        config.TRAINING_ARGS["per_device_train_batch_size"] = args.batch_size
        
    if args.max_steps > 0:
        config.TRAINING_ARGS["max_steps"] = args.max_steps
        # Ensure we save at the end
        config.TRAINING_ARGS["save_steps"] = args.max_steps
        
    train(limit=args.limit)

if __name__ == "__main__":
    main()
