"""
Script to compare baseline and improved evaluation results.
"""
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluate import compute_exact_match

def load_results(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return []
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_metrics(results, name):
    if not results:
        return f"{name}: No results found.\n"
    
    total = len(results)
    exact_matches = 0
    
    for item in results:
        prediction = item.get('generated_query', '')
        reference = item.get('gold_query', '')
        
        if compute_exact_match(prediction, reference):
            exact_matches += 1
            
    accuracy = (exact_matches / total) * 100
    return f"{name}: {total} examples. Exact Match Accuracy: {accuracy:.2f}% ({exact_matches}/{total})\n"

def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    baseline_file = os.path.join(base_dir, 'baseline_results.json')
    improved_file = os.path.join(base_dir, 'improved_results.json')
    output_file = os.path.join(base_dir, 'comparison.txt')
    
    baseline_results = load_results(baseline_file)
    improved_results = load_results(improved_file)
    
    # We will invoke evaluate.py functionality here if possible
    
    output_file = os.path.join(base_dir, 'comparison.txt')
    
    baseline_summary = calculate_metrics(baseline_results, "Baseline")
    improved_summary = calculate_metrics(improved_results, "Improved (Fine-tuned)")
    
    with open(output_file, 'w') as f:
        f.write("Evaluation Comparison\n")
        f.write("=====================\n\n")
        f.write(baseline_summary)
        f.write(improved_summary)
        
        # Detailed Analysis on Intersection
        f.write("\nDetailed Analysis (Intersection of first {} examples):\n".format(min(len(baseline_results), len(improved_results))))
        f.write("---------------------------------------------------\n")
        
        limit = min(len(baseline_results), len(improved_results))
        for i in range(limit):
            b_item = baseline_results[i]
            i_item = improved_results[i]
            
            # Use normalize_sql or just simple string match for now (imports available in evaluate.py)
            b_correct = compute_exact_match(b_item.get('generated_query', ''), b_item.get('gold_query', ''))
            i_correct = compute_exact_match(i_item.get('generated_query', ''), i_item.get('gold_query', ''))
            
            if b_correct != i_correct:
                status = "IMPROVED" if i_correct else "REGRESSED"
                f.write(f"\n[Example {i+1}] Status: {status}\n")
                f.write(f"Question: {b_item.get('question')}\n")
                f.write(f"Gold: {b_item.get('gold_query')}\n")
                f.write(f"Baseline (0.5B): {b_item.get('generated_query')} ({'Correct' if b_correct else 'Incorrect'})\n")
                f.write(f"Improved (1.5B): {i_item.get('generated_query')} ({'Correct' if i_correct else 'Incorrect'})\n")

        f.write("\nNote: Execution accuracy requires database access and was not computed in this summary.\n")
        
    print(f"Comparison saved to {output_file}")
    print(baseline_summary)
    print(improved_summary)

if __name__ == "__main__":
    main()
