"""
Download and extract the Spider dataset from Google Drive.

This script automates the process of downloading the Spider 1.0 dataset
and extracting it to the data/spider directory.
"""

import os
import sys
import zipfile
import gdown

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Constants
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
SPIDER_URL = f"https://drive.google.com/uc?id={SPIDER_GDRIVE_ID}"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SPIDER_DIR = os.path.join(DATA_DIR, 'spider')
ZIP_PATH = os.path.join(DATA_DIR, 'spider.zip')


def download_spider() -> None:
    """
    Download the Spider dataset from Google Drive.
    
    Raises:
        Exception: If download fails
    """
    print(f"Downloading Spider dataset from Google Drive...")
    print(f"This may take a few minutes...")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download the dataset
    try:
        gdown.download(SPIDER_URL, ZIP_PATH, quiet=False)
        print(f"✓ Download complete: {ZIP_PATH}")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        raise


def extract_spider() -> None:
    """
    Extract the Spider dataset ZIP file.
    
    Raises:
        Exception: If extraction fails
    """
    print(f"\nExtracting Spider dataset...")
    
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"✓ Extraction complete: {SPIDER_DIR}")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        raise
    
    # Clean up ZIP file
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print(f"✓ Cleaned up ZIP file")


def validate_dataset() -> bool:
    """
    Validate that the Spider dataset was downloaded and extracted correctly.
    
    Returns:
        True if validation passes, False otherwise
    """
    print(f"\nValidating dataset structure...")
    
    required_files = [
        os.path.join(SPIDER_DIR, 'train.json'),
        os.path.join(SPIDER_DIR, 'dev.json'),
        os.path.join(SPIDER_DIR, 'tables.json'),
        os.path.join(SPIDER_DIR, 'database'),
    ]
    
    all_valid = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {os.path.basename(file_path)}")
        else:
            print(f"✗ Missing: {os.path.basename(file_path)}")
            all_valid = False
    
    if all_valid:
        # Count databases
        db_dir = os.path.join(SPIDER_DIR, 'database')
        if os.path.isdir(db_dir):
            databases = [d for d in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, d))]
            print(f"✓ Found {len(databases)} databases")
    
    return all_valid


def main() -> None:
    """
    Main entry point for downloading the Spider dataset.
    """
    print("=" * 60)
    print("Spider Dataset Downloader")
    print("=" * 60)
    
    # Check if dataset already exists
    if os.path.exists(SPIDER_DIR):
        response = input(f"\nSpider dataset already exists at {SPIDER_DIR}\nRe-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            if validate_dataset():
                print("\n✓ Dataset is valid and ready to use!")
            return
    
    try:
        # Download and extract
        download_spider()
        extract_spider()
        
        # Validate
        if validate_dataset():
            print("\n" + "=" * 60)
            print("✓ Spider dataset successfully downloaded and validated!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✗ Dataset validation failed. Please check the files.")
            print("=" * 60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
