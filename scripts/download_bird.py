import os
import requests
import zipfile
import shutil
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
BIRD_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
ZIP_PATH = DATA_DIR / "minidev.zip"
EXTRACT_DIR = DATA_DIR / "bird"

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded to {dest_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")
        exit(1)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def main():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)
    
    if not ZIP_PATH.exists():
        download_file(BIRD_URL, ZIP_PATH)
    
    if not EXTRACT_DIR.exists():
        EXTRACT_DIR.mkdir()
    
    extract_zip(ZIP_PATH, EXTRACT_DIR)
    
    # Check what we have
    print("Download and extraction complete.")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for name in files:
            print(os.path.join(root, name))

if __name__ == "__main__":
    main()
