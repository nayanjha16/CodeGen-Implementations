
import glob
import json
import os

# List input files correctly sorted
files = sorted(glob.glob('data/training/temp_out_*.json'))
print(f"Merging {len(files)} files...")

final_data = []
success_count = 0
fail_count = 0

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            chunk = json.load(fh)
            for item in chunk:
                if 'mql_pipeline' in item:
                    final_data.append(item)
                    success_count += 1
                else:
                    fail_count += 1
    except Exception as e:
        print(f"Error reading {f}: {e}")

output_path = "data/training/bird_train_with_mql_CLEAN.json"

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"DONE. Merged {success_count} items. (Filtered {fail_count} failures).")
print(f"Output: {output_path}")
