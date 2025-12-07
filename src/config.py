"""
Central configuration for the Text-to-SQL project.
"""
import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

# Model Configuration
BASELINE_MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct" 
IMPROVED_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Default to Improved for Training context, but explicit elsewhere
MODEL_NAME = IMPROVED_MODEL_NAME 

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT_QUANTIZATION = True if DEVICE == "cuda" else False  # Set to True to use QLoRA/4-bit quantization (saves VRAM)

# Dataset Configuration
SPIDER_DIR = os.path.join(DATA_DIR, 'spider')

# Training Configuration
TRAINING_ARGS = {
    "output_dir": os.path.join(OUTPUT_DIR, 'checkpoints'),
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "logging_steps": 10,
    "save_steps": 100,
    "fp16": True if DEVICE == "cuda" else False,
    "optim": "paged_adamw_8bit" if DEVICE == "cuda" else "adamw_torch",  # Use paged optimizer for QLoRA, regular for others
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
