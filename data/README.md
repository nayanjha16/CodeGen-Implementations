# Spider Dataset

This directory contains the Spider benchmark dataset for Text-to-SQL evaluation.

## About Spider

Spider is a large-scale, complex, and cross-domain semantic parsing and text-to-SQL dataset. It consists of 10,181 questions and 5,693 unique complex SQL queries on 200 databases across 138 different domains.

- **Paper**: [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887)
- **Website**: https://yale-lily.github.io/spider
- **License**: CC BY-SA 4.0

## Dataset Structure

After downloading, the `spider/` directory will contain:

```
spider/
├── train.json          # Training examples (7,000 examples)
├── dev.json            # Development/test examples (1,034 examples)
├── tables.json         # Database schema definitions
├── database/           # SQLite database files
│   ├── concert_singer/
│   │   └── concert_singer.sqlite
│   ├── car_1/
│   │   └── car_1.sqlite
│   └── ... (200 databases total)
└── README.txt
```

## Downloading the Dataset

Run the download script:

```bash
python scripts/download_spider.py
```

This will:
1. Download the Spider dataset from Google Drive (~90 MB compressed)
2. Extract it to `data/spider/`
3. Validate the dataset structure

## Dataset Format

### Example Entry (from `dev.json`):

```json
{
  "db_id": "concert_singer",
  "question": "How many singers do we have?",
  "query": "SELECT count(*) FROM singer"
}
```

### Fields:
- **db_id**: Database identifier (matches folder name in `database/`)
- **question**: Natural language question
- **query**: Ground truth SQL query

### Tables Format (`tables.json`):

Contains schema information for each database:
- `db_id`: Database identifier
- `table_names_original`: List of table names
- `column_names_original`: List of (table_idx, column_name) pairs
- `column_types`: Data types for each column
- `primary_keys`: Primary key column indices
- `foreign_keys`: Foreign key relationships

## Using the Dataset

### Load Examples

```python
from src.dataset_loader import load_spider_dataset

# Load development set
examples = load_spider_dataset(split='dev')

# Load with limit (for testing)
examples = load_spider_dataset(split='dev', limit=10)

# Access example data
for ex in examples:
    print(ex.question)      # Natural language question
    print(ex.query)         # Ground truth SQL
    print(ex.db_id)         # Database ID
    print(ex.db_path)       # Path to SQLite file
```

### Get Database Schema

```python
from src.dataset_loader import get_database_schema

# Get CREATE TABLE statements for a database
schema = get_database_schema('concert_singer')
print(schema)
```

### Run Evaluation

```bash
# Evaluate on full dev set
python scripts/run_eval.py

# Evaluate on subset (for quick testing)
python scripts/run_eval.py --limit 10

# Skip execution accuracy (faster)
python scripts/run_eval.py --skip-execution

# Verbose output (show each example)
python scripts/run_eval.py --limit 5 --verbose
```

## Evaluation Metrics

### Exact Match (EM)
Compares normalized SQL strings (case-insensitive, whitespace-normalized).

### Execution Accuracy (EX)
Executes both predicted and reference queries on the actual SQLite database and compares results.

## Database Access

Each database is a SQLite file located at:
```
data/spider/database/{db_id}/{db_id}.sqlite
```

You can query them directly using Python's `sqlite3` module:

```python
import sqlite3

conn = sqlite3.connect('data/spider/database/concert_singer/concert_singer.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT * FROM singer LIMIT 5")
results = cursor.fetchall()
conn.close()
```

## Citation

If you use the Spider dataset, please cite:

```bibtex
@inproceedings{yu2018spider,
  title={Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={3911--3921},
  year={2018}
}
```

## Troubleshooting

### Dataset not found
Make sure you've run `python scripts/download_spider.py` first.

### Database file missing
Verify the download completed successfully and the database directory exists.

### Permission errors
Ensure you have write permissions in the `data/` directory.
