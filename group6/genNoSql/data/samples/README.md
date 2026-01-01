# Sample Data

This directory contains synthetic MongoDB databases and queries for testing the Text-to-NoSQL system.

## Files

- `ecommerce.json`: E-commerce database (Products, Customers) with 10 sample queries
- `school.json`: School database (Students, Courses) with 5 sample queries
- `training_examples.json`: Combined training examples from all datasets

## Usage

To load this data into the system:

```python
from src.utils.sample_generator import SampleDataGenerator

generator = SampleDataGenerator()
generator.save_sample_data("data/samples")
```

To use with MongoDB:

```python
from src.utils.mongo_client import MongoDBClient
import json

# Load sample data
with open("data/samples/ecommerce.json") as f:
    data = json.load(f)

# Create MongoDB collections
client = MongoDBClient()
client.connect()

for coll_name, coll_data in data["databases"].items():
    client.create_collection(coll_name, coll_data["documents"])
```

## Dataset Statistics

- **E-commerce**: 2 collections, 6 documents total, 10 queries
- **School**: 1 collection, 3 documents, 5 queries
- **Total Training Examples**: 15 queries

## Query Types Covered

- Simple find() with filters
- Projections
- Aggregations ($group, $avg, $sum)
- $unwind for nested arrays
- $regex for text matching
- distinct() queries
- countDocuments()
- sort() and limit()
