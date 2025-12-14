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
    execution_matches = 0
    
    for item in results:
        # Use pre-calculated metrics if available, otherwise compute (fallback)
        if 'exact_match' in item:
            if item['exact_match']:
                exact_matches += 1
        else:
            # Fallback
            if compute_exact_match(item.get('generated_query', ''), item.get('gold_query', '')):
                exact_matches += 1

        if 'execution_match' in item:
            if item['execution_match']:
                execution_matches += 1
        # No fallback for execution accuracy as we don't have DB path handy easily here
            
    em_accuracy = (exact_matches / total) * 100
    ex_accuracy = (execution_matches / total) * 100
    
    return (f"{name}: {total} examples.\n"
            f"  Exact Match Accuracy: {em_accuracy:.2f}% ({exact_matches}/{total})\n"
            f"  Execution Accuracy:   {ex_accuracy:.2f}% ({execution_matches}/{total})\n")

def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    baseline_file = os.path.join(base_dir, 'baseline_results.json')
    improved_file = os.path.join(base_dir, 'improved_results.json')
    output_file = os.path.join(base_dir, 'comparison.txt')
    
    baseline_results = load_results(baseline_file)
    improved_results = load_results(improved_file)
    
    output_file = os.path.join(base_dir, 'comparison.txt')
    
    baseline_summary = calculate_metrics(baseline_results, "Baseline")
    improved_summary = calculate_metrics(improved_results, "Improved (Fine-tuned)")
    
    with open(output_file, 'w') as f:
        f.write("Evaluation Comparison\n")
        f.write("=====================\n\n")
        f.write(baseline_summary)
        f.write("\n")
        f.write(improved_summary)
        
        # Detailed Analysis on Intersection
        f.write("\nDetailed Analysis (Intersection of first {} examples):\n".format(min(len(baseline_results), len(improved_results))))
        f.write("---------------------------------------------------\n")
        
        limit = min(len(baseline_results), len(improved_results))
        for i in range(limit):
            b_item = baseline_results[i]
            i_item = improved_results[i]
            
            # Use 'exact_match' field or recompute
            b_correct = b_item.get('exact_match', False)
            i_correct = i_item.get('exact_match', False)
            
            b_exec = b_item.get('execution_match', False)
            i_exec = i_item.get('execution_match', False)

            if b_correct != i_correct or b_exec != i_exec:
                status = "IMPROVED" if (i_correct and not b_correct) or (i_exec and not b_exec) else "REGRESSED" if (b_correct and not i_correct) or (b_exec and not i_exec) else "CHANGED"
                
                f.write(f"\n[Example {i+1}] Status: {status}\n")
                f.write(f"Question: {b_item.get('question')}\n")
                f.write(f"Complexity: {b_item.get('complexity', 'unknown')}\n")
                f.write(f"Baseline: {b_item.get('generated_query')}\n")
                f.write(f"  Exact Match: {'[PASS]' if b_correct else '[FAIL]'}, Execution: {'[PASS]' if b_exec else '[FAIL]'}\n")
                f.write(f"Improved: {i_item.get('generated_query')}\n")
                f.write(f"  Exact Match: {'[PASS]' if i_correct else '[FAIL]'}, Execution: {'[PASS]' if i_exec else '[FAIL]'}\n")

        f.write("\nEnd of Comparison\n")
        
    print(f"Comparison saved to {output_file}")
    print(baseline_summary)
    print(improved_summary)

if __name__ == "__main__":
    main()
