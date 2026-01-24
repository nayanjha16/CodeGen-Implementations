
import requests
import zipfile
import os
from tqdm import tqdm

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train_databases.zip"
ZIP_PATH = "data/training/train_databases.zip"
EXTRACT_DIR = "data/training/train_databases"

def download_file(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to connect: {response.status_code}")
        return False
        
    total_size = int(response.headers.get('content-length', 0))
    
    if os.path.exists(filename):
        if os.path.getsize(filename) == total_size:
            print("File already downloaded.")
            return True
        else:
            print("Incomplete file found. Redownloading...")
            
    with open(filename, 'wb') as f, tqdm(
        desc="Downloading Databases",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    return True

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Done extraction.")

if __name__ == "__main__":
    if not os.path.exists("data/training"):
        os.makedirs("data/training")
        
    print(f"Checking URL: {URL}")
    if download_file(URL, ZIP_PATH):
        unzip_file(ZIP_PATH, EXTRACT_DIR)
        print("SUCCESS: Databases ready.")
    else:
        print("FAILURE: Could not download.")
