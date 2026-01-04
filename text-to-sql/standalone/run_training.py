"""
================================================================================
TEXT-TO-SQL TRAINING (QLoRA) - ALL-IN-ONE SCRIPT
================================================================================

This is a consolidated version of the fine-tuning pipeline using QLoRA.
All code is in one file for easy understanding of the flow.

WHAT IS QLoRA?
- LoRA: Low-Rank Adaptation - adds small trainable matrices to frozen model
- QLoRA: Quantized LoRA - uses 4-bit quantization to reduce memory

FLOW:
1. Configuration (line ~40)
2. Load Spider Training Dataset (line ~80)
3. Format Data for Training (line ~180)
4. Load Model with Quantization (line ~230)
5. Configure LoRA Adapters (line ~290)
6. Training Loop (line ~340)
7. Save Model (line ~400)

To run:
    uv run python standalone/run_training.py --limit 100 --dry-run
    uv run python standalone/run_training.py --epochs 3
    
REQUIREMENTS:
    - GPU with CUDA recommended (CPU works but very slow)
    - ~8GB VRAM for 1.5B model with 4-bit quantization
    
================================================================================
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    TaskType
)
from trl import SFTTrainer


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SPIDER_DIR = os.path.join(DATA_DIR, 'spider')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'checkpoints')

# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Model to fine-tune

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT_QUANTIZATION = DEVICE == "cuda"  # Only use quantization on GPU

# LoRA Configuration
# These are the key hyperparameters for efficient fine-tuning
LORA_CONFIG = {
    "r": 16,              # Rank - lower = fewer trainable params, faster
    "lora_alpha": 32,     # Scaling factor for LoRA weights
    "lora_dropout": 0.05, # Dropout for regularization
    "bias": "none",       # Don't train bias terms
    "target_modules": [   # Which layers to apply LoRA to
        "q_proj",         # Query projection in attention
        "k_proj",         # Key projection in attention
        "v_proj",         # Value projection in attention
        "o_proj",         # Output projection in attention
        "gate_proj",      # MLP gate
        "up_proj",        # MLP up projection
        "down_proj"       # MLP down projection
    ]
}

# Training Configuration
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch = 4 * 4 = 16
    "learning_rate": 2e-4,
    "logging_steps": 10,
    "save_steps": 100,
    "fp16": DEVICE == "cuda",
    "optim": "paged_adamw_8bit" if DEVICE == "cuda" else "adamw_torch"
}

print(f"""
================================================================================
TRAINING CONFIGURATION
================================================================================
Model: {MODEL_NAME}
Device: {DEVICE}
4-bit Quantization: {USE_4BIT_QUANTIZATION}
LoRA Rank: {LORA_CONFIG['r']}
Learning Rate: {TRAINING_CONFIG['learning_rate']}
Batch Size: {TRAINING_CONFIG['per_device_train_batch_size']}
Output Dir: {OUTPUT_DIR}
================================================================================
""")


# ==============================================================================
# SECTION 2: DATASET LOADING
# ==============================================================================

class SpiderExample:
    """Represents a single example from the Spider dataset."""
    
    def __init__(self, question: str, query: str, db_id: str):
        self.question = question
        self.query = query
        self.db_id = db_id


def load_spider_dataset(split: str = 'train', limit: Optional[int] = None) -> List[SpiderExample]:
    """Load Spider training dataset."""
    json_filename = 'train_spider.json' if split == 'train' else f'{split}.json'
    json_path = os.path.join(SPIDER_DIR, json_filename)
    
    # Try alternative filename
    if not os.path.exists(json_path):
        json_path = os.path.join(SPIDER_DIR, 'train.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found: {json_path}")
    
    print(f"Loading Spider {split} dataset from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        example = SpiderExample(
            question=item['question'],
            query=item['query'],
            db_id=item['db_id']
        )
        examples.append(example)
        
        if limit and len(examples) >= limit:
            break
    
    print(f"Loaded {len(examples)} training examples")
    return examples


def load_spider_tables() -> Dict[str, Any]:
    """Load database schemas from tables.json."""
    tables_path = os.path.join(SPIDER_DIR, 'tables.json')
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    return {table['db_id']: table for table in tables}


def get_database_schema(db_id: str) -> str:
    """Generate CREATE TABLE statements for a database."""
    tables = load_spider_tables()
    
    if db_id not in tables:
        return ""  # Return empty if not found
    
    db_info = tables[db_id]
    schema_lines = []
    
    table_names = db_info['table_names_original']
    column_names = db_info['column_names_original']
    column_types = db_info['column_types']
    
    tables_columns = {}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:
            continue
        table_name = table_names[table_idx]
        if table_name not in tables_columns:
            tables_columns[table_name] = []
        col_type = column_types[col_idx]
        tables_columns[table_name].append(f"  {col_name} {col_type}")
    
    for table_name, columns in tables_columns.items():
        schema_lines.append(f"CREATE TABLE {table_name} (")
        schema_lines.extend(columns)
        schema_lines.append(")")
        schema_lines.append("")
    
    return "\n".join(schema_lines)


# ==============================================================================
# SECTION 3: DATA FORMATTING FOR TRAINING
# ==============================================================================

def format_training_example(example: SpiderExample, schema: str) -> Dict[str, Any]:
    """
    Format a single example into Qwen chat format.
    
    The model learns to generate SQL (assistant response) given 
    the schema and question (user message).
    
    Format:
        User: [schema + question]
        Assistant: [SQL query]
    """
    
    user_content = f"""You are a SQL query generator. Given a database schema and question, output ONLY the SQL query.

DATABASE SCHEMA:
{schema}

QUESTION: {example.question}

SQL:"""

    # Qwen chat format
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example.query}  # Target SQL
    ]
    
    return {"messages": messages}


def prepare_training_data(examples: List[SpiderExample]) -> List[Dict[str, Any]]:
    """
    Prepare all training examples.
    
    Each example becomes a conversation:
    - User provides schema + question
    - Assistant responds with SQL
    """
    formatted_data = []
    
    print("Formatting training data...")
    for example in tqdm(examples, desc="Formatting"):
        try:
            schema = get_database_schema(example.db_id)
            if schema:  # Only include if schema found
                formatted = format_training_example(example, schema)
                formatted_data.append(formatted)
        except Exception as e:
            continue  # Skip failed examples
    
    print(f"Formatted {len(formatted_data)} examples for training")
    return formatted_data


# ==============================================================================
# SECTION 4: MODEL LOADING WITH QUANTIZATION
# ==============================================================================

def load_model_for_training():
    """
    Load model with optional 4-bit quantization.
    
    4-bit quantization:
    - Reduces memory by ~4x
    - Allows training larger models on smaller GPUs
    - Uses BitsAndBytes library
    """
    print(f"Loading model: {MODEL_NAME}...")
    
    if USE_4BIT_QUANTIZATION:
        print("Using 4-bit quantization (QLoRA configuration)...")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Use 4-bit precision
            bnb_4bit_quant_type="nf4",            # Normal Float 4-bit
            bnb_4bit_compute_dtype=torch.float16, # Compute in fp16
            bnb_4bit_use_double_quant=True,       # Double quantization
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",  # Automatic GPU placement
            trust_remote_code=True
        )
        
        # Prepare for k-bit training (required for QLoRA)
        model = prepare_model_for_kbit_training(model)
        
    else:
        print(f"Loading model on {DEVICE} (no quantization)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True
        )
        if DEVICE == "cpu":
            model = model.to("cpu")
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Model loaded successfully")
    return model, tokenizer


# ==============================================================================
# SECTION 5: LORA CONFIGURATION
# ==============================================================================

def apply_lora(model):
    """
    Apply LoRA adapters to the model.
    
    LoRA (Low-Rank Adaptation):
    - Freezes original model weights
    - Adds small trainable matrices to specific layers
    - Only trains ~1% of parameters
    - Much faster and memory-efficient than full fine-tuning
    
    Mathematical formulation:
        Original: y = Wx
        With LoRA: y = Wx + (A @ B)x
        Where A and B are small trainable matrices
    """
    print("Configuring LoRA adapters...")
    
    peft_config = LoraConfig(
        r=LORA_CONFIG["r"],                      # Rank of decomposition
        lora_alpha=LORA_CONFIG["lora_alpha"],    # Scaling factor
        lora_dropout=LORA_CONFIG["lora_dropout"],# Dropout
        bias=LORA_CONFIG["bias"],                # Bias training
        task_type=TaskType.CAUSAL_LM,            # Task type
        target_modules=LORA_CONFIG["target_modules"]  # Layers to adapt
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    # Example output: "trainable params: 6,553,600 || all params: 1,543,714,816 || trainable%: 0.42%"
    
    return model, peft_config


# ==============================================================================
# SECTION 6: TRAINING LOOP
# ==============================================================================

def train(
    limit: Optional[int] = None,
    epochs: int = 3,
    batch_size: int = 4,
    max_steps: int = -1,
    dry_run: bool = False
):
    """
    Main training function.
    
    Args:
        limit: Limit number of training examples
        epochs: Number of training epochs
        batch_size: Batch size per device
        max_steps: Maximum steps (-1 for full training)
        dry_run: If True, run only 1 step for testing
    """
    
    print("=" * 60)
    print("STARTING FINE-TUNING")
    print("=" * 60)
    
    # Override config for dry run
    if dry_run:
        print("DRY RUN MODE - Running 1 step only")
        max_steps = 1
        epochs = 1
    
    # Step 1: Load and format dataset
    train_examples = load_spider_dataset(split='train', limit=limit)
    formatted_data = prepare_training_data(train_examples)
    dataset = Dataset.from_list(formatted_data)
    
    # Step 2: Load model
    model, tokenizer = load_model_for_training()
    
    # Step 3: Apply LoRA
    model, peft_config = apply_lora(model)
    
    # Step 4: Configure training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        logging_steps=TRAINING_CONFIG["logging_steps"] if not dry_run else 1,
        save_steps=TRAINING_CONFIG["save_steps"] if not dry_run else 1,
        fp16=TRAINING_CONFIG["fp16"],
        optim=TRAINING_CONFIG["optim"],
        report_to="none",  # Don't report to wandb/tensorboard
        push_to_hub=False,
        use_cpu=DEVICE == "cpu",
    )
    
    # Step 5: Create trainer
    # SFTTrainer is from TRL (Transformer Reinforcement Learning) library
    # Specifically designed for supervised fine-tuning
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # Step 6: Train!
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    
    trainer.train()
    
    # Step 7: Save the trained LoRA adapter
    final_checkpoint_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    print(f"\nSaving model to: {final_checkpoint_path}")
    trainer.save_model(final_checkpoint_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Adapter saved to: {final_checkpoint_path}")
    print("To use the fine-tuned model, load the adapter with PeftModel.from_pretrained()")


# ==============================================================================
# SECTION 7: MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Text-to-SQL model with QLoRA")
    parser.add_argument("--limit", type=int, default=None, help="Limit training examples")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 step only (testing)")
    
    args = parser.parse_args()
    
    train(
        limit=args.limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

