# 🧠 Text-to-SQL Project - Deep Dive

## **1. The Big Picture**

This project converts **natural language questions** into **SQL queries** using a Small Language Model (SLM). For example:

| Input (Natural Language) | Output (SQL) |
|--------------------------|--------------|
| "Show me all users older than 25" | `SELECT * FROM users WHERE age > 25;` |

---

## **2. Architecture Overview**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TEXT-TO-SQL PIPELINE                              │
│                                                                          │
│   User Query ──► RAG Retriever ──► Prompt Builder ──► SLM ──► SQL       │
│       │              │                  │              │        │       │
│   "Find users    Uses semantic      Combines       Qwen 0.5B  Generated │
│    over 25"      similarity to      schema +       generates  SQL query │
│                  find relevant      question       tokens               │
│                  database schema    into prompt                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## **3. Core Components Explained**

### **📦 Model Loader (`model_loader.py`)**
- Loads **Qwen2.5-Coder-0.5B-Instruct** from HuggingFace
- This is a **500 million parameter** coding-specialized model
- Runs on **GPU (CUDA)** if available, otherwise **CPU**
- Auto-downloads on first run (~988MB)

```python
model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
```

### **🔍 RAG - Retrieval Augmented Generation (`rag.py`)**
Uses **SentenceTransformers** (`all-MiniLM-L6-v2`) for semantic search:

1. **Initialization**: Converts all database schemas into vector embeddings
2. **Retrieval**: When a query comes in:
   - Converts the query to a vector
   - Finds the most similar schema using **cosine similarity**
   - Returns the relevant schema as context

**Why RAG?** The model can't know about your specific database tables. RAG provides the schema context dynamically.

### **🎯 Inference (`inference.py` & `inference_improved.py`)**
The core SQL generation logic:

1. **Baseline (`inference.py`)**: Uses the pre-trained model directly.
2. **Improved (`inference_improved.py`)**: Uses the **fine-tuned adapter** and enhanced prompting/post-processing.

**Prompt Construction** - Uses a structured format:
   ```
   ### Instruction:
   You are a text-to-SQL generator...
   
   ### Schema:
   CREATE TABLE users (id INT, name TEXT, age INT)
   
   ### Question:
   Show me all users older than 25
   
   ### Response:
   ```

2. **Token Generation** - Model generates SQL token by token
3. **Post-processing** - Cleans up the output (removes markdown, extra statements)

### **📊 Few-Shot Inference (`inference_fewshot.py`)**
An enhanced version that includes **example pairs** in the prompt:
- Shows 3 example (schema → question → SQL) pairs
- Helps the model understand the expected output format
- Generally produces more reliable results

### **📁 Dataset Loader (`dataset_loader.py`)**
Loads the **Spider benchmark dataset** for evaluation:
- `SpiderExample` class: Holds question, ground-truth SQL, and database info
- `load_spider_dataset()`: Loads train/dev split
- `get_database_schema()`: Builds CREATE TABLE statements from tables.json

### **📈 Evaluator (`evaluate.py`)**
Two key metrics for measuring quality:

| Metric | Description |
|--------|-------------|
| **Exact Match** | Does the predicted SQL string match the reference (after normalization)? |
| **Execution Accuracy** | Do both queries return the **same results** when executed on the database? |

Execution accuracy is more meaningful because multiple SQL queries can produce the same result.

### **🏋️ Fine-tuning (`train.py`)**
Allows adapting the model to the specific Spider dataset style:
- Uses **QLoRA** (Quantized Low-Rank Adaptation) for efficient training
- **Dataset Formatter**: Converts Spider examples to Qwen chat format
- **Trainer**: Uses HuggingFace `SFTTrainer` with PEFT
- **Benefits**: Improves complex query generation and schema understanding

---

## **4. Data Flow - Step by Step**

```
1. User enters: "Show me all users who are older than 25"
                    │
                    ▼
2. RAG RETRIEVAL ───────────────────────────────────────────
   ├─ Encode query → 384-dimensional vector
   ├─ Compare with all schema embeddings
   └─ Return: "CREATE TABLE users (id INT, name TEXT, age INT, email TEXT)"
                    │
                    ▼
3. PROMPT CONSTRUCTION ─────────────────────────────────────
   │  "### Instruction: You are a text-to-SQL generator...
   │   ### Schema: CREATE TABLE users (...)
   │   ### Question: Show me all users who are older than 25
   │   ### Response:"
                    │
                    ▼
4. TOKENIZATION ────────────────────────────────────────────
   │  Convert text → Token IDs: [2309, 4521, 892, ...]
                    │
                    ▼
5. MODEL INFERENCE (Qwen2.5-Coder) ─────────────────────────
   │  24 Transformer layers process the input
   │  Generate new tokens one at a time
   │  Output: [2309, 4521, 892, ..., 7823, 1294, ...]
                    │
                    ▼
6. DETOKENIZATION ──────────────────────────────────────────
   │  Token IDs → Text: "SELECT * FROM users WHERE age > 25;"
                    │
                    ▼
7. POST-PROCESSING ─────────────────────────────────────────
   │  Remove markdown formatting
   │  Extract only the SQL part
   │  Keep only first statement
                    │
                    ▼
8. OUTPUT: "SELECT * FROM users WHERE age > 25;"
```

---

## **5. Evaluation Pipeline**

The evaluation scripts (`run_eval.py` and `run_improved_eval.py`) do:

1. **Load Spider Dataset** (1034 dev examples)
2. **For each example:**
   - Get the database schema
   - Generate SQL using the model (Baseline or Fine-tuned)
   - Compute exact match (string comparison)
   - Compute execution accuracy (run both queries, compare results)
3. **Report metrics:**
   - Exact Match Accuracy: X%
   - Execution Accuracy: Y%

We compare the results of the **Baseline** vs. **Improved** models to measure the impact of fine-tuning.

---

## **6. Technology Stack**

| Component | Technology |
|-----------|------------|
| **SLM** | Qwen2.5-Coder-0.5B-Instruct (HuggingFace) |
| **Embeddings** | all-MiniLM-L6-v2 (SentenceTransformers) |
| **ML Framework** | PyTorch, Transformers |
| **Database** | SQLite (Spider dataset) |
| **Dataset** | Spider 1.0 Benchmark |

---

## **7. Key Design Decisions**

1. **Small Model (0.5B params)**: Runs on consumer hardware, fast inference
2. **RAG for Context**: Dynamic schema retrieval vs. fine-tuning on specific databases
3. **Instruction-tuned Model**: Follows structured prompts well
4. **Zero-shot + Few-shot**: Both approaches available for different use cases
5. **Spider Benchmark**: Standard dataset for reproducible evaluation

---

## **8. Project Structure Summary**

```
text-to-sql/
├── src/
│   ├── model_loader.py    # Load Qwen model from HuggingFace
│   ├── rag.py             # Semantic search for schema retrieval
│   ├── inference.py       # Zero-shot SQL generation
│   ├── inference_fewshot.py # Few-shot SQL generation
│   ├── dataset_loader.py  # Spider dataset utilities
│   └── evaluate.py        # Exact match & execution accuracy
├── scripts/
│   ├── run_eval.py        # Run baseline evaluation
│   ├── run_improved_eval.py # Run improved evaluation
│   └── download_spider.py # Download Spider dataset
├── docs/
│   └── sequence_diagrams.md # Detailed Mermaid sequence diagrams
└── data/spider/           # Dataset (gitignored)
```

---

## **9. How to Run**

### Interactive Demo
```bash
python scripts/run_inference.py
```

### Evaluation on Spider Dataset
```bash
# First, download the dataset
python scripts/download_spider.py

# Then run evaluation
python scripts/run_eval.py --limit 100  # Test on 100 examples
python scripts/run_eval.py              # Full evaluation
```

---

## **10. Understanding the Model**

### Qwen2.5-Coder-0.5B-Instruct

- **Architecture**: Transformer-based decoder-only model
- **Parameters**: 494 million
- **Context Length**: 32,768 tokens
- **Specialization**: Code generation and understanding
- **Training**: Instruction-tuned for following prompts

The model processes input through:
1. **Embedding Layer**: Converts tokens to vectors
2. **24 Transformer Layers**: Each with self-attention and feed-forward networks
3. **Output Layer**: Projects to vocabulary size for next-token prediction

---

## **11. RAG Deep Dive**

### Why RAG?

Without RAG, the model would need to:
- Be fine-tuned on every possible database schema
- Hallucinate table/column names it doesn't know

With RAG:
- Schema is retrieved dynamically at runtime
- Model receives exact table structure as context
- Works with any database without retraining

### Embedding Model

**all-MiniLM-L6-v2**:
- Produces 384-dimensional vectors
- Fast and lightweight (~22M parameters)
- Good semantic understanding of text

---

## **12. Evaluation Metrics Explained**

### Exact Match (EM)

```python
# Normalization steps:
# 1. Lowercase
# 2. Remove extra whitespace
# 3. Remove trailing semicolons

predicted = "SELECT * FROM users WHERE age > 25"
reference = "select * from users where age > 25"
# After normalization: MATCH ✓
```

### Execution Accuracy (EX)

```python
# Different SQL, same result:
predicted = "SELECT * FROM users WHERE age > 25"
reference = "SELECT * FROM users WHERE 25 < age"

# Both return same rows → Execution Match ✓
```

---

This project demonstrates a **complete end-to-end Text-to-SQL pipeline** using modern NLP techniques, suitable for both learning and practical applications.

---

## **13. Results and Analysis/Comparison**

### **Baseline Model (0.5B)**
- **Configuration**: Qwen2.5-Coder-0.5B-Instruct (Zero-shot)
- **Execution**: Successful on CPU/GPU.
- **Performance**:
  - Sample Size: 100 examples
  - Time per example: ~5.4s (CPU)
  - Accuracy: 20-30% (exact match range on small subset)

### **Improved Model (1.5B)**
- **Configuration**: Qwen2.5-Coder-1.5B-Instruct (Fine-tuned / Improved Prompt)
- **Status**: **FAILED** on current Hardware (CPU Only).
- **Failure Analysis**:
  1.  **Fine-tuning**: The 1.5B model requires significantly more RAM/VRAM than available. Training attempts failed with memory errors (OOM) or immediate process termination.
  2.  **Inference (Evaluation)**: Zero-shot inference was attempted as a fallback.
      - **Latency**: ~105s per example (vs 5s for 0.5B).
      - **Stability**: Process stalled/timed out after 40% completion (limit 10).
- **Conclusion**:
  - The 0.5B model is highly efficient for CPU-based local development.
  - The 1.5B model provides stronger reasoning but **strictly requires GPU acceleration** (16GB+ VRAM recommended for fine-tuning) for practical usage.

