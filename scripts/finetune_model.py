
import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
NEW_MODEL = "Qwen2.5-Coder-3B-Instruct-mql-adapter"
DATA_PATH = "data/training/sql_to_mql_finetuning.jsonl"

def main():
    print(f"Loading model: {MODEL_NAME}")

    # Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for FP16 training with Flash Attention

    # Load Data
    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # Format function for SFT
    # Qwen uses ChatML format. 
    # We need to format the "messages" into a single string for training.
    def format_chat_template(row):
        # Apply chat template
        # "messages" is a list of dicts.
        # We use the tokenizer's apply_chat_template
        
        # Note: SFTTrainer expects a text field if formatting_func is provided
        # or we map it beforehand.
        
        conversation = row["messages"]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    print("Formatting dataset...")
    dataset = dataset.map(format_chat_template)
    
    # LoRA Config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Training Params
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2, # Adjust based on VRAM (Safe for 8GB)
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none"
    )

    print("Starting Training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2048, # Supports up to 32k, but 2k is enough for SQL/MQL usually
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    trainer.train()

    print("Training Complete. Saving Model...")
    trainer.model.save_pretrained(NEW_MODEL)
    tokenizer.save_pretrained(NEW_MODEL)
    print(f"Model saved to {NEW_MODEL}")

if __name__ == "__main__":
    main()
