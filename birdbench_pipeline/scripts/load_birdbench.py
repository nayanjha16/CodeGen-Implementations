import os
import yaml
from datasets import load_dataset

# Load paths config
with open("configs/paths.yaml") as f:
    paths = yaml.safe_load(f)

RAW_DIR = paths["raw_birdbench"]
os.makedirs(RAW_DIR, exist_ok=True)

def load_birdbench():
    """
    Loads BirdBench Mini-Dev dataset from HuggingFace
    and saves it locally for reproducibility.
    """
    print("Downloading BirdBench Mini-Dev dataset...")
    
    dataset = load_dataset("birdsql/bird_mini_dev")

    # Save JSON locally
    dataset_path = os.path.join(RAW_DIR, "birdbench_mini_dev")
    dataset.save_to_disk(dataset_path)

    print(f"BirdBench Mini-Dev saved to: {dataset_path}")
    return dataset

if __name__ == "__main__":
    load_birdbench()
