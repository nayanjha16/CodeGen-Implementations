
import json
import os
import sys

def check_bird_size():
    # Helper to find where data might be
    possible_paths = [
        "data/bird/minidev/dev.json", 
        "data/bird/dev.json",
        "data/bird/train.json", 
        "data/bird/minidev/train.json"
    ]
    
    found_path = None
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if not found_path:
        # Search recursively
        for root, dirs, files in os.walk("data"):
            for f in files:
                if f.endswith(".json") and ("dev" in f or "train" in f):
                    found_path = os.path.join(root, f)
                    break
            if found_path: break
            
    if not found_path:
        print("Could not find any BirdBench JSON file.")
        return

    print(f"Inspecting file: {found_path}")
    try:
        with open(found_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = len(data)
        print(f"Total examples: {count}")
        
        target_start = 4250
        target_end = 5668
        
        if count < target_start:
            print(f"WARNING: Dataset is too small for requested range {target_start}-{target_end}.")
        else:
            available_end = min(count, target_end)
            print(f"Range {target_start}-{available_end} is available.")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_bird_size()
