# Text-to-SQL

A Text-to-SQL system that converts natural language questions into SQL queries using RAG (Retrieval-Augmented Generation) and LLMs.

## Features

- 🔍 **RAG-based Schema Retrieval**: Automatically retrieves relevant database schema information
- 🤖 **SLM-powered SQL Generation**: Uses Qwen2.5-Coder-0.5B-Instruct to generate accurate SQL queries
- 🚀 **Fine-tuning Support**: Improve performance by fine-tuning on the Spider dataset using QLoRA
- 📊 **Spider Dataset Support**: Integrated with the Spider benchmark dataset for evaluation
- 📈 **Evaluation Metrics**: Supports Exact Match and Execution Accuracy metrics

## Project Structure

## Project Structure

```
text-to-sql/
├── data/                    # Dataset directory (gitignored)
│   └── spider/              # Spider dataset files
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
│   ├── download_spider.py   # Download Spider dataset
│   ├── run_eval.py          # Run baseline evaluation
│   ├── run_improved_eval.py # Run improved evaluation (fine-tuned)
│   ├── compare_results.py   # Compare baseline vs improved results
│   └── run_finetuning.py    # Run fine-tuning
├── src/                     # Source code
│   ├── config.py            # Central configuration
│   ├── dataset_loader.py    # Dataset loading utilities
│   ├── dataset_formatter.py # Dataset formatting for training
│   ├── evaluate.py          # Evaluation logic
│   ├── inference.py         # Inference logic
│   ├── inference_improved.py # Improved inference with fuzzy matching
│   ├── model_loader.py      # Model loading utilities
│   ├── rag.py               # RAG implementation
│   ├── train.py             # Training logic
│   └── utils.py             # Helper utilities
├── colab_runner.ipynb       # Jupyter Notebook for Google Colab
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended) or CPU
- ~2GB disk space for model weights

### Installation (Local)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mr-Rachapudi/text-to-sql.git
   cd text-to-sql
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Spider dataset**
   
   The Spider dataset is not included in the repository due to its size (~1GB). Run the download script:
   ```bash
   python scripts/download_spider.py
   ```
   
   This will download and extract the Spider 1.0 dataset to `data/spider/`.

5. **Model Download (Automatic)**
   
   The Qwen2.5-Coder-0.5B-Instruct model will be automatically downloaded from HuggingFace on first run.

### ☁️ Google Colab Setup (Recommended for 1.5B Model)

Since the larger improved model requires a GPU, we provide a notebook for running evaluations on Google Colab (Free Tier T4 GPU).

1. **Download the project**: Download this repository as a ZIP file (or clone and zip the `text-to-sql` folder).
2. **Open Colab**: Go to [Google Colab](https://colab.research.google.com/).
3. **Upload Notebook**: Upload `colab_runner.ipynb`.
4. **Enable GPU**: Go to `Runtime` > `Change runtime type` > Select `T4 GPU`.
5. **Upload Project Zip**: Follow the notebook instructions to upload your `text-to-sql.zip`.
6. **Run**: Execute the cells to set up the environment and run the evaluation.

## Usage

### Run Evaluation

Run baseline evaluation (pre-trained model):
```bash
python scripts/run_eval.py [--limit 100] [--complex-only]
```

Run improved evaluation (fine-tuned model):
```bash
python scripts/run_improved_eval.py [--limit 100] [--complex-only]
```

Compare results:
```bash
python scripts/compare_results.py
```

### Run Fine-tuning

You can fine-tune the model on the Spider dataset to improve performance.

```bash
# Run fine-tuning (requires GPU)
python scripts/run_finetuning.py --epochs 3 --batch-size 4

# Run a quick test (dry run)
python scripts/run_finetuning.py --dry-run
```

The fine-tuning script uses **QLoRA** (Quantized Low-Rank Adaptation) to train efficiently with minimal memory usage. Checkpoints are saved to `results/checkpoints`.

## Dataset

This project uses the [Spider](https://yale-lily.github.io/spider) dataset, a large-scale complex and cross-domain semantic parsing and text-to-SQL dataset.

**Note**: The dataset files are excluded from version control via `.gitignore` due to their size. Use `scripts/download_spider.py` to obtain the dataset.

## License

MIT License
