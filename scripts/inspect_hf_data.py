
from datasets import load_dataset

print("Loading dataset info...")
ds = load_dataset("birdsql/bird", trust_remote_code=True)
print("Keys:", ds.keys())
print("Train Features:", ds['train'].features)
print("First Item Keys:", ds['train'][0].keys())

# Check for schema/tables
if 'tables' in ds['train'][0]:
    print("Found 'tables' column!")
    print(ds['train'][0]['tables'])
elif 'schema' in ds['train'][0]:
    print("Found 'schema' column!")
    print(ds['train'][0]['schema'])
else:
    print("No obvious schema column found.")
