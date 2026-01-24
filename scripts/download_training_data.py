
import os
import json
from datasets import load_dataset
from tqdm import tqdm

def download_bird_training_data():
    """Downloads the BIRD training dataset and saves it as JSON."""
    
    output_dir = "data/training"
    output_file = os.path.join(output_dir, "bird_train.json")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading birdsql/bird23-train-filtered dataset...")
    try:
        # Load the dataset
        dataset = load_dataset("birdsql/bird23-train-filtered", split="train")
        
        print(f"Dataset downloaded. Total examples: {len(dataset)}")
        
        # Convert to list of dicts for JSON serialization
        data_list = []
        for item in tqdm(dataset, desc="Processing examples"):
            data_list.append({
                "question": item.get("question"),
                "sql": item.get("SQL"),
                "db_id": item.get("db_id"),
                "evidence": item.get("evidence"), # Helper context if available
            })
            
        # Save to JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
            
        print(f"SUCCESS: Saved {len(data_list)} training examples to {output_file}")
        
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")

if __name__ == "__main__":
    download_bird_training_data()
