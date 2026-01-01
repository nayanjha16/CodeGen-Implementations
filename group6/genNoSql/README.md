# Text-to-NoSQL Translation System

Implementation of the **TEND (Text-to-NoSQL Dataset)** paper's **SMART Framework** (SLM-assisted and RAG-assisted Multi-step framework) for converting natural language queries into MongoDB NoSQL queries.

## 📋 Overview

This system implements a sophisticated 4-step pipeline for translating natural language questions into executable MongoDB queries:

1. **Schema Prediction**: Predicts relevant collections and fields from the natural language query
2. **Query Generation**: Generates initial MongoDB query using LLM
3. **RAG Refinement**: Refines query using similar examples from training data
4. **Execution Optimization**: Executes query and debugs errors automatically

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Natural Language Query                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Schema Prediction (SLM/LLM)                       │
│  → Predicts relevant collections and fields                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Query Generation (SLM/LLM)                        │
│  → Generates initial MongoDB query                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: RAG Refinement                                     │
│  → Retrieves similar examples                               │
│  → Refines query using LLM + examples                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Execution Optimization                             │
│  → Executes query on MongoDB                                │
│  → Debugs and fixes errors automatically                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              MongoDB Query Results
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- MongoDB (local or Atlas cloud)
- OpenAI API key (or other LLM provider)

### Setup

1. **Clone and navigate to the project**:
```bash
cd /Users/vparupati/CapStone/textToNosql
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. **Configure MongoDB** (optional):
Edit `configs/config.yaml` to set MongoDB connection string.

## 🚀 Quick Start

### Demo Mode

Run the demo with sample e-commerce data:

```bash
python src/main.py --mode demo
```

This will:
- Set up sample MongoDB collections (Products, Customers)
- Run several example queries
- Show the complete pipeline for each query

### Interactive Mode

Query the system interactively:

```bash
python src/main.py --mode interactive
```

Example queries:
- "Find all products in the Electronics category"
- "Get customers from New York"
- "What is the average price of products?"

### Evaluation Mode

Evaluate on a test dataset:

```bash
python src/main.py --mode eval --test-file data/samples/ecommerce.json --output-dir results
```

## 📊 Dataset

### Using Your Own Data

**Place TEND Dataset** (if you have it):
```bash
# Copy your TEND dataset to:
cp -r /path/to/tend/data data/raw/tend
```

**Use Converted Spider Data**:
```bash
# Copy your converted Spider data to:
cp -r /path/to/converted/spider data/processed/tend
```

### Sample Data

Sample e-commerce and school databases are included in `data/samples/`:
- `ecommerce.json`: Products and Customers collections
- `school.json`: Students collection
- `training_examples.json`: 15 example queries for RAG

Generate more samples:
```python
from src.utils.sample_generator import SampleDataGenerator

generator = SampleDataGenerator()
generator.save_sample_data("data/samples")
```

## ⚙️ Configuration

Edit `configs/config.yaml`:

```yaml
# MongoDB Configuration
mongodb:
  uri: "mongodb://localhost:27017/"  # or MongoDB Atlas URI
  database_name: "tend_db"

# Model Configuration
models:
  llm_provider: "openai"  # openai, anthropic, local
  llm_model: "gpt-4"
  temperature: 0.0

# RAG Configuration
rag:
  enabled: true
  num_examples: 3
  nlq_weight: 0.4
  query_weight: 0.3
  schema_weight: 0.3
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v
```

Test specific components:
```bash
# Test MongoDB connection
pytest tests/test_mongo_client.py

# Test SMART pipeline
pytest tests/test_smart_pipeline.py
```

## 📚 Usage Examples

### Python API

```python
from src.smart.smart_pipeline import create_smart_framework

# Initialize framework
framework = create_smart_framework("configs/config.yaml")
framework.connect_mongodb()

# Load schemas
schemas = {
    "Products": ["_id", "name", "category", "price", "stock"],
    "Customers": ["_id", "name", "email", "city"]
}
framework.load_schemas(schemas)

# Translate query
result = framework.translate("Find all products in Electronics category")

print(f"Generated Query: {result['final_query']}")
print(f"Results: {result['results']}")
```

### With RAG

```python
# Index training examples for better results
training_examples = [
    {
        "question": "Find all products",
        "query": "db.Products.find({});",
        "schema": "Collection: Products\nFields: _id, name"
    },
    # ... more examples
]

framework.index_training_examples(training_examples)

# Now queries will use similar examples for refinement
result = framework.translate("Show me all customers")
```

## 📈 Evaluation Metrics

The system implements metrics from the TEND paper:

1. **Exact Match (EM)**: Query string exact match
2. **Component Match**: Collection, operation, filter matching
3. **Execution Accuracy (EX)**: Do queries return same results?
4. **Valid Execution (VE)**: Does query execute without errors?

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(mongo_client)

metrics = evaluator.evaluate_single(
    predicted_query="db.Products.find({});",
    gold_query="db.Products.find({});"
)

print(f"Exact Match: {metrics['exact_match']}")
print(f"Execution Accuracy: {metrics['execution_accuracy']}")
```

## 🔬 Advanced Features

### Fine-tuned Models

To use fine-tuned SLMs instead of LLM APIs:

1. Train models on TEND dataset
2. Save to `models/schema_predictor/` and `models/query_generator/`
3. Update config:

```yaml
models:
  use_fine_tuned: true
  schema_predictor_path: "models/schema_predictor"
  query_generator_path: "models/query_generator"
```

### Custom LLM Provider

Implement `BaseLLMClient` for custom providers:

```python
from src.utils.llm_client import BaseLLMClient

class CustomLLMClient(BaseLLMClient):
    def generate(self, prompt, system_prompt=None, **kwargs):
        # Your implementation
        pass
```

## 📁 Project Structure

```
textToNosql/
├── data/
│   ├── raw/              # Raw dataset (Spider, TEND)
│   ├── processed/        # Processed NoSQL databases
│   └── samples/          # Sample data for testing
├── src/
│   ├── smart/            # SMART framework components
│   │   ├── schema_predictor.py
│   │   ├── query_generator.py
│   │   ├── rag_refiner.py
│   │   ├── execution_optimizer.py
│   │   └── smart_pipeline.py
│   ├── utils/            # Utilities
│   │   ├── mongo_client.py
│   │   ├── llm_client.py
│   │   ├── embeddings.py
│   │   └── sample_generator.py
│   ├── evaluation/       # Evaluation metrics
│   │   └── evaluator.py
│   └── main.py           # Main entry point
├── configs/
│   └── config.yaml       # Configuration
├── notebooks/
│   └── demo.ipynb        # Demo notebook
└── tests/                # Unit tests
```

## 🛠️ Troubleshooting

### MongoDB Connection Issues

**Error: Connection refused**
```bash
# Start local MongoDB
brew services start mongodb-community

# Or use MongoDB Atlas cloud URI in config
```

**Error: Authentication failed**
```bash
# Update config.yaml with correct credentials
mongodb:
  uri: "mongodb+srv://username:password@cluster.mongodb.net/"
```

### LLM API Issues

**Error: API key not found**
```bash
# Ensure .env file has your API key
OPENAI_API_KEY=your_key_here
```

**Error: Rate limit exceeded**
```bash
# Add delay between requests or use exponential backoff
# Or switch to Claude/local models
```

## 📖 References

- **TEND Paper**: "Bridging the Gap: Enabling Natural Language Queries for NoSQL Databases through Text-to-NoSQL Translation"
- **Spider Dataset**: Large-scale semantic parsing benchmark
- **MongoDB Aggregation**: [Official Documentation](https://docs.mongodb.com/manual/aggregation/)

## 🤝 Contributing

This is a research implementation. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 👥 Authors

Based on the TEND paper implementation for the CapStone project.

## 🙏 Acknowledgments

- TEND paper authors for the framework design
- Spider dataset creators
- MongoDB community for excellent documentation
