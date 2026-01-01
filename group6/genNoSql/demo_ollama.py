#!/usr/bin/env python3
"""
Demo script for Text-to-NoSQL with Ollama and MongoDB Docker
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from smart.smart_pipeline import create_smart_framework
from utils.sample_generator import SampleDataGenerator
import json

print("=" * 70)
print(" Text-to-NoSQL Demo with Ollama (Llama 3.1:8b) + MongoDB Docker")
print("=" * 70)
print()

# Check Ollama is running
print("🔍 Checking Ollama...")
try:
    import requests
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        print("✅ Ollama is running")
        models = response.json().get("models", [])
        llama_models = [m for m in models if "llama3" in m["name"]]
        if llama_models:
            print(f"✅ Found Llama 3 models: {[m['name'] for m in llama_models]}")
        else:
            print("⚠️  No Llama 3 models found")
    else:
        print("❌ Ollama is not responding correctly")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error connecting to Ollama: {e}")
    print("Please start Ollama: ollama serve")
    sys.exit(1)

print()

# Initialize framework
print("🚀 Initializing SMART framework with Ollama...")
framework = create_smart_framework("configs/config_ollama.yaml")

# Connect to MongoDB
print("🔌 Connecting to MongoDB (Docker container)...")
if framework.connect_mongodb():
    print("✅ Connected to MongoDB successfully")
else:
    print("❌ Failed to connect to MongoDB")
    print("Make sure MongoDB container is running:")
    print("  docker run -d --name text2nosql-mongo -p 27017:27017 mongo:7.0")
    sys.exit(1)

print()

# Load sample data
print("📊 Setting up sample data...")
sample_file = "data/samples/ecommerce.json"

if not os.path.exists(sample_file):
    print("Generating sample data...")
    generator = SampleDataGenerator()
    generator.save_sample_data()

with open(sample_file) as f:
    data = json.load(f)

# Create MongoDB collections
databases = data["databases"]
for coll_name, coll_data in databases.items():
    print(f"  Creating collection: {coll_name}")
    framework.mongo_client.create_collection(coll_name, coll_data["documents"])

print("✅ Sample data loaded")
print()

# Load schemas
print("📋 Loading database schemas...")
framework.load_schemas(data["schema_summary"])
print(f"✅ Loaded schemas for: {list(framework.schemas.keys())}")
print()

# Index training examples
print("🔍 Indexing training examples for RAG...")
training_file = "data/samples/training_examples.json"
if os.path.exists(training_file):
    with open(training_file) as f:
        examples = json.load(f)
    framework.index_training_examples(examples)
    print(f"✅ Indexed {len(examples)} training examples")
else:
    print("⚠️  No training examples found, RAG will be less effective")

print()
print("=" * 70)
print(" Running Demo Queries with Llama 3.1:8b")
print("=" * 70)
print()

# Demo queries
demo_queries = [
    "Find all products in the Electronics category",
    "Count the total number of products",
    "Find customers who live in New York",
]

for i, nlq in enumerate(demo_queries, 1):
    print(f"\n{'='*70}")
    print(f"Query {i}/{len(demo_queries)}")
    print(f"{'='*70}")
    print(f"📝 Natural Language: {nlq}")
    print()
    
    try:
        # Translate (without execution optimization to be faster)
        print("🤖 Processing with SMART framework...")
        result = framework.translate(nlq, use_execution_optimization=False)
        
        # Display results
        print()
        print("Step 1 - Schema Prediction:")
        pred_schema = result['steps']['schema_prediction']
        print(f"  Collection: {pred_schema.get('collection', 'N/A')}")
        print(f"  Fields: {pred_schema.get('fields', [])}")
        
        print()
        print("Step 2 - Initial Query:")
        print(f"  {result['steps']['initial_query']}")
        
        print()
        print("Step 3 - RAG-Refined Query:")
        print(f"  {result['steps']['refined_query']}")
        
        print()
        print("🎯 Final Query:")
        print(f"  {result['final_query']}")
        
        # Try to execute
        if result['final_query']:
            print()
            print("▶️  Executing query...")
            exec_result = framework.mongo_client.execute_query(result['final_query'])
            
            if exec_result['success']:
                results = exec_result['results']
                print(f"✅ Query executed successfully!")
                print(f"   Returned {len(results)} document(s)")
                
                if results:
                    print()
                    print("   Sample results:")
                    for j, doc in enumerate(results[:2], 1):
                        print(f"   {j}. {doc}")
            else:
                print(f"❌ Execution failed: {exec_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print()
print("=" * 70)
print(" Demo Complete!")
print("=" * 70)
print()
print("✅ Successfully demonstrated Text-to-NoSQL with:")
print("   - Ollama (Llama 3.1:8b) for LLM inference")
print("   - MongoDB Docker container for data storage")
print("   - SMART framework (4-step pipeline)")
print("   - RAG-enhanced query generation")
print()
print("📚 Next steps:")
print("   - Try interactive mode: python src/main.py --mode interactive --config configs/config_ollama.yaml")
print("   - Add your own data to data/raw/")
print("   - Fine-tune on TEND dataset for better performance")
print()

# Cleanup
framework.disconnect()
print("🔌 Disconnected from MongoDB")
