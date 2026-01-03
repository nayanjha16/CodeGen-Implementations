import os

from utils.gemini_client import GeminiClient
from utils.spider_dataset import load_fixed_spider
from evaluation.sql_accuracy import normalize_sql, token_accuracy

def main():
    # Init Gemini client
    print("Initializing Gemini client...")
    client = GeminiClient()
    print("Gemini client ready.")
    # Load dataset
    print("Loading Turbular/fixed_spider dataset from HuggingFace...")
    dataset = load_fixed_spider(split="train")
    print(f"Loaded {len(dataset)} examples.")
    # 3. Use a small subset for demo (e.g. first 5 examples)
    #num_examples = 5
    num_examples = 10
    exact_match_count = 0   # <-- MUST be here
    total = 0 
    print(f"\nGenerating SQL for first {num_examples} examples...\n")    
    print

    for idx in range(num_examples):
        example = dataset[idx]
        question = example["question"]
        db_id = example["db"]
        gold_sql = example["query"]
        db_schema = example["db_schema"]

        print("=" * 80)
        print(f"Example #{idx}")
        print(f"DB ID    : {db_id}")
        print(f"Question : {question}")
        print("-" * 80)
        print("Ground truth SQL from dataset:")
        print(gold_sql)
        print("-" * 80)

        try:
            predicted_sql = client.generate_sql(
                question=question,
                db_schema=db_schema,
                gold_sql=gold_sql,
             )
        except Exception as e:
            print(f"[ERROR] Gemini call failed: {e}")
            continue
        total += 1

        print("Gemini predicted SQL:")
        print(predicted_sql)
        print("=" * 80)
        print()

        print("Predicted SQL vs Gold SQL:")
        score = token_accuracy(predicted_sql, gold_sql) # calculate token accuracy
        print("Token accuracy:", score)   

if __name__ == "__main__":
    main()
