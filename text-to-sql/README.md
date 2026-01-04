# Text-to-SQL Minimal Implementation

A minimal, self-contained Text-to-SQL implementation using Qwen2.5-Coder models with QLoRA fine-tuning.

## 🏗️ Project Structure

```
text-to-sql-minimal/
├── pyproject.toml              # UV package manager config
├── README.md                   # This file
├── standalone/                 # ALL-IN-ONE scripts (no external imports)
│   ├── run_baseline_evaluation.py    # Baseline: 0.5B model, zero-shot
│   ├── run_improved_evaluation.py    # Improved: 1.5B model + LoRA adapter
│   └── run_training.py               # QLoRA fine-tuning script
├── scripts/
│   ├── download_spider.py      # Download Spider dataset
│   └── compare_results.py      # Compare baseline vs improved results
├── data/
│   └── spider/                 # Spider dataset (download separately)
└── results/
    ├── baseline_results.json   # Baseline evaluation results
    ├── improved_results.json   # Improved evaluation results
    └── checkpoints/            # Fine-tuned model weights
```

## 🚀 Quick Start

### 1. Install UV Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Download Spider Dataset
```bash
uv run python scripts/download_spider.py
# If extracted to data/spider_data, move contents:
mv data/spider_data/* data/spider/
```

### 3. Run Baseline Evaluation (0.5B model, zero-shot)
```bash
uv run python standalone/run_baseline_evaluation.py --limit 20
```

### 4. Run Improved Evaluation (1.5B model, with/without fine-tuning)
```bash
# Without LoRA adapter (zero-shot)
uv run python standalone/run_improved_evaluation.py --limit 20 --no-adapter

# With LoRA adapter (after training)
uv run python standalone/run_improved_evaluation.py --limit 20
```

### 5. Fine-tune with QLoRA
```bash
uv run python standalone/run_training.py
```

### 6. Compare Results
```bash
uv run python scripts/compare_results.py
cat results/comparison.txt
```

## 📊 Expected Results (100 examples)

| Model Configuration | Exact Match | Execution Accuracy | Time |
|---------------------|-------------|-------------------|------|
| 0.5B Baseline (zero-shot) | ~15% | ~45% | ~3 min |
| 1.5B No Adapter (zero-shot) | ~30% | ~45% | ~21 min |
| 1.5B + QLoRA (100 ex, 1 epoch) | ~18% | ~32% | ~21 min |

> **Note**: QLoRA with limited data (100 examples, 1 epoch) may cause overfitting to complex patterns. For better results, train with 500+ examples and 3+ epochs.

## 🔧 Standalone Scripts Overview

### `run_baseline_evaluation.py`
- **Model**: Qwen2.5-Coder-0.5B-Instruct (494M params)
- **Mode**: Zero-shot (no fine-tuning)
- **Key Features**: Simple prompt engineering

### `run_improved_evaluation.py`
- **Model**: Qwen2.5-Coder-1.5B-Instruct (1.5B params)
- **Mode**: Optional LoRA adapter loading
- **Key Features**:
  - Fuzzy column correction (fixes hallucinated column names)
  - Better stop conditions (prevents rambling)
  - Post-processing to clean SQL output

### `run_training.py`
- **Technique**: QLoRA (4-bit quantization + LoRA)
- **Dataset**: Spider training set
- **Output**: LoRA adapter weights in `results/checkpoints/`

## 📝 Key Concepts

### QLoRA Fine-tuning
- Trains only ~0.1% of parameters using Low-Rank Adaptation
- Uses 4-bit quantization to reduce memory usage
- Adapts the model to SQL generation task

### Fuzzy Column Correction
```python
# If model generates: SELECT petage FROM pets
# And schema has:      pet_age
# Corrects to:         SELECT pet_age FROM pets
```

### Evaluation Metrics
- **Exact Match**: Generated SQL == Gold SQL (after normalization)
- **Execution Accuracy**: Generated SQL returns same results as Gold SQL

## 🔬 Customization

### Adjust Training Parameters
Edit `standalone/run_training.py`:
```python
TRAIN_LIMIT = 500      # Number of training examples
NUM_EPOCHS = 3         # Training epochs
LORA_R = 16           # LoRA rank
LORA_ALPHA = 32       # LoRA alpha
```

### Adjust Evaluation
```bash
# Evaluate only complex queries
uv run python standalone/run_baseline_evaluation.py --complex-only --limit 50

# Evaluate with specific limit
uv run python standalone/run_improved_evaluation.py --limit 100
```

## 📚 References

- [Spider Dataset](https://yale-lily.github.io/spider)
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [PEFT/LoRA](https://huggingface.co/docs/peft)
- [TRL SFTTrainer](https://huggingface.co/docs/trl)

