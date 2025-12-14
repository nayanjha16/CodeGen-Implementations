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

from src.evaluate import compute_exact_match, compute_execution_accuracy

def main():
    parser = argparse.ArgumentParser(description="Run Improved Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--smart", action="store_true", help="Use smart filtering (75 mixed, 25 complex)")
    parser.add_argument("--complex-only", action="store_true", help="Evaluate only complex queries")
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
    # If complex-only is requested, load all first then filter
    load_limit = args.limit if not args.complex_only else None
    examples = load_spider_dataset(split='dev', limit=load_limit, smart_filter=args.smart)
    
    if args.complex_only:
        examples = [e for e in examples if e.complexity == 'complex']
        if args.limit:
            examples = examples[:args.limit]
            
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
            
        exact_match = compute_exact_match(generated_sql, example.query)
        execution_match = False
        if example.db_path and os.path.exists(example.db_path):
            execution_match = compute_execution_accuracy(generated_sql, example.query, example.db_path)
            
        results.append({
            "question": example.question,
            "gold_query": example.query,
            "generated_query": generated_sql,
            "db_id": example.db_id,
            "exact_match": exact_match,
            "execution_match": execution_match,
            "complexity": getattr(example, 'complexity', 'unknown')
        })
        
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Improved results saved to {output_file}")

if __name__ == "__main__":
    main()
