
import glob
import json
import os

files = sorted(glob.glob('data/training/temp_out_*.json'))
print(f"Found {len(files)} chunk files.")

total_converted = 0
total_items = 0

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            converted = sum(1 for item in data if 'mql_pipeline' in item)
            total = len(data)
            print(f"{os.path.basename(f)}: {converted} / {total} converted")
            total_converted += converted
            total_items += total
    except Exception as e:
        print(f"{os.path.basename(f)}: ERROR {e}")

print(f"TOTAL: {total_converted} / {total_items}")
