import sys
import os
import json
import argparse
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_loader import load_model
from src.inference_improved import generate_sql_improved
from src.dataset_loader import load_spider_dataset, get_database_schema
import src.config as config

def main():
    parser = argparse.ArgumentParser(description="Run Improved Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    args = parser.parse_args()

    print("Running Improved Evaluation (Fine-tuned)...")
    
    # Path to adapter
    adapter_path = os.path.join(config.TRAINING_ARGS["output_dir"], "final_checkpoint")
    if not os.path.exists(adapter_path):
        print(f"Warning: Adapter not found at {adapter_path}. Running in Zero-Shot mode with Base Model.")
        adapter_path = None
    
    # Load Model (Adapter optional)
    model, tokenizer = load_model(adapter_path=adapter_path, model_name=config.IMPROVED_MODEL_NAME)
    
    # Load Dev Set
    examples = load_spider_dataset(split='dev', limit=args.limit)
    print(f"Loaded {len(examples)} examples for evaluation.")
    
    results = []
    
    output_file = os.path.join(config.OUTPUT_DIR, 'improved_results.json')
    
    for example in tqdm(examples, desc="Evaluating"):
        schema = get_database_schema(example.db_id)
        
        try:
            generated_sql = generate_sql_improved(model, tokenizer, example.question, schema)
        except Exception as e:
            print(f"Error generating SQL for {example.question}: {e}")
            generated_sql = "SELECT * FROM error"
            
        results.append({
            "question": example.question,
            "gold_query": example.query,
            "generated_query": generated_sql,
            "db_id": example.db_id
        })
        
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Improved results saved to {output_file}")

if __name__ == "__main__":
    main()
