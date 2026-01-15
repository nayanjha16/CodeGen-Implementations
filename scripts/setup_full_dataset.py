
import os
import zipfile
from pathlib import Path

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total files for progress
            total = len(zip_ref.infolist())
            print(f"Found {total} files in archive.")
            
            # Extract
            zip_ref.extractall(extract_to)
            print("Extraction complete.")
            return True
    except Exception as e:
        print(f"Failed to extract {zip_path}: {e}")
        return False

def main():
    base_data_dir = Path("data")
    bird_dir = base_data_dir / "bird"
    bird_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Train Set
    train_zip = base_data_dir / "train.zip"
    if train_zip.exists():
        print(f"Found {train_zip}")
        extract_zip(train_zip, bird_dir)
    else:
        print(f"Warning: {train_zip} not found.")

    # 2. Dev Set
    dev_zip = base_data_dir / "dev.zip"
    if dev_zip.exists():
        print(f"Found {dev_zip}")
        extract_zip(dev_zip, bird_dir)
    else:
        print(f"Warning: {dev_zip} not found.")

    print("\nDataset setup logic finished. Check data/bird for extracted files.")

if __name__ == "__main__":
    main()
