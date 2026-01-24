
import os
import json
import subprocess
import math
import argparse
from concurrent.futures import ThreadPoolExecutor

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_worker(key, part_idx, total_parts, sleep_time):
    input_file = f"data/training/temp_part_{part_idx}.json"
    output_file = f"data/training/temp_out_{part_idx}.json"
    
    # HARDCODED PAID KEY for every worker
    REAL_API_KEY = "AIzaSyChH7ygjUBycMrv-WxxuLOE3znAtdtwuHE"
    
    print(f"[{part_idx}] Starting Worker {part_idx+1}/{total_parts}...")
    
    cmd = [
        "python", "scripts/convert_sql_to_mql.py",
        "--input", input_file,
        "--output", output_file,
        "--api_key", REAL_API_KEY,
        "--sleep", str(sleep_time),
        "--model", "gemini-2.0-flash"
    ]
    
    with open(f"data/training/log_part_{part_idx}.txt", "w") as log_f:
        subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        
    print(f"[{part_idx}] Worker finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", required=True, help="Dummy file path")
    parser.add_argument("--input", default="data/training/bird_train.json")
    parser.add_argument("--output", default="data/training/bird_train_with_mql.json")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep time per worker")
    args = parser.parse_args()
    
    # FORCE 8 WORKERS
    num_workers = 8
    dummy_keys = ["PAID_KEY"] * num_workers
    
    print(f"Forcing {num_workers} parallel workers using Gemini 2.0 Flash...")
    
    # Load and Split Data
    data = load_data(args.input)
    chunk_size = math.ceil(len(data) / num_workers)
    
    for i in range(num_workers):
        start = i * chunk_size
        end = start + chunk_size
        chunk = data[start:end]
        
        # Save chunk
        with open(f"data/training/temp_part_{i}.json", "w", encoding='utf-8') as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
            
    print(f"Split {len(data)} items into {num_workers} chunks of ~{chunk_size} items.")
    
    # Run Parallel Workers
    print("Launching workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, key in enumerate(dummy_keys):
            futures.append(executor.submit(run_worker, key, i, num_workers, args.sleep))
            
        for f in futures:
            f.result()
            
    print("All workers done. Merging results...")
    
    # Merge
    final_data = []
    for i in range(num_workers):
        outfile = f"data/training/temp_out_{i}.json"
        if os.path.exists(outfile):
            with open(outfile, 'r', encoding='utf-8') as f:
                final_data.extend(json.load(f))
        else:
            print(f"WARN: Output file {outfile} missing!")
            
    # Save Final
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
        
    print(f"SUCCESS: Total {len(final_data)} items saved to {args.output}")

if __name__ == "__main__":
    main()
