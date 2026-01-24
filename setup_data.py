import os
import requests
import zipfile
import io

DATA_DIR = "data"
REPO_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"

def setup_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Check if already extracted (simple check)
    # if os.path.exists(os.path.join(DATA_DIR, "mini_dev")):
    #    print("Dataset already appears to be present.")
    #    return

    print(f"Downloading dataset from {REPO_ZIP_URL}...")
    try:
        r = requests.get(REPO_ZIP_URL)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print("Extracting...")
        z.extractall(DATA_DIR)
        print("Done!")
        
        # Renaissance of folder structure if needed
        # The zip will create 'mini_dev-main', we might want to rename or just use it.
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    setup_data()
