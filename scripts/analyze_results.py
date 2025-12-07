import json

# Load results
with open('evaluation_results.json', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("BASELINE EVALUATION RESULTS")
print("=" * 80)
print()
print(f"Total Examples:        {data['metrics']['total_examples']}")
print(f"Exact Match Count:     {data['metrics']['exact_match_count']}")
print(f"Exact Match Accuracy:  {data['metrics']['exact_match_accuracy']:.2f}%")
print(f"Execution Count:       {data['metrics']['execution_count']}")
print(f"Execution Accuracy:    {data['metrics']['execution_accuracy']:.2f}%")
print()
print("=" * 80)
print("SAMPLE FAILURES (First 5)")
print("=" * 80)

failures = [r for r in data['detailed_results'] if not r['execution_match']]

for i, failure in enumerate(failures[:5]):
    print(f"\nExample {failure['index'] + 1}:")
    print(f"  Question:  {failure['question']}")
    print(f"  Database:  {failure['db_id']}")
    print(f"  Reference: {failure['reference_sql']}")
    print(f"  Predicted: {failure['predicted_sql']}")
    print()
