# Raw Data Directory

This directory is for storing your raw datasets (TEND or converted Spider data).

## Option 1: TEND Dataset

If you have access to the TEND dataset, copy it here:

```bash
# Copy your TEND dataset
cp -r /path/to/tend/dataset/* /Users/vparupati/CapStone/textToNosql/data/raw/
```

Expected structure:
```
data/raw/
├── databases/          # MongoDB database JSON files
├── queries/            # NoSQL query files
└── questions/          # Natural language questions
```

## Option 2: Converted Spider Dataset

If you have converted Spider data using the Java tool mentioned in the paper:

```bash
# Copy your converted Spider data
cp -r /path/to/converted/spider/* /Users/vparupati/CapStone/textToNosql/data/raw/spider/
```

Expected structure:
```
data/raw/spider/
├── train_spider.json   # Training examples
├── dev.json            # Development set
└── tables.json         # Table schemas
```

## Option 3: Use Sample Data

If you don't have the TEND or Spider data yet, the system includes sample synthetic data for testing:

```bash
# Sample data is already generated in:
data/samples/
├── ecommerce.json
├── school.json
└── training_examples.json
```

You can use the sample data to:
- Test the system immediately
- Understand the data format
- Develop and debug without access to full datasets

## Data Format

### Training Examples JSON Format

```json
[
  {
    "question": "Natural language query",
    "query": "db.collection.find({});",
    "schema": "Collection: collection\nFields: field1, field2",
    "collection": "collection",
    "predicted_fields": "field1, field2"
  }
]
```

### Database Schema Format

```json
{
  "CollectionName": ["field1", "field2", "field3"]
}
```

## Next Steps

After copying your data:

1. **Update configuration** in `configs/config.yaml`:
   ```yaml
   data:
     spider_path: "data/raw/spider"
     tend_path: "data/processed/tend"
   ```

2. **Process the data** (if needed):
   ```python
   # If you need to transform SQL to NoSQL
   from src.transformation.db_transformer import DBTransformer
   transformer = DBTransformer()
   transformer.process_spider_data()
   ```

3. **Run the system**:
   ```bash
   python src/main.py --mode demo
   ```

## Notes

- The system works with or without the full TEND/Spider datasets
- Sample data is sufficient for initial testing and development
- For production use, more training examples improve RAG performance
- Minimum recommended: 50-100 training examples for decent RAG retrieval
