
"""
Fine-tuning script specifically for BirdBench Text-to-SQL.
"""
import os
import sys
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

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_loader import load_bird_dataset, get_bird_schema
from dataset_formatter import prepare_training_data

def train_bird(limit=None, force=False):
    print(f"Starting BIRD-SQL fine-tuning for {config.MODEL_NAME}...")
    
    # 0. Output Directory
    output_dir = os.path.join("results", "bird_finetune")
    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    
    if os.path.exists(final_output_dir) and not force:
        print(f"✓ Checkpoint already exists at {final_output_dir}")
        return
    
    # 1. Load Dataset
    print("Loading BirdBench dataset (Training Split, Range 4250-5668)...")
    # Tries to load full train set first, falls back to what is available
    bird_examples = load_bird_dataset(split='train', slice_range=(4250, 5668), limit=limit)
    
    if not bird_examples:
        print("Train set not found. Falling back to Mini-Dev (Dev Split)...")
        bird_examples = load_bird_dataset(split='dev', limit=limit)

    if not bird_examples:
        print("No BirdBench examples found (checked Train and Mini-Dev). Aborting.")
        return
        
    print(f"Loaded {len(bird_examples)} examples for training.")
    
    # Format data 
    # Logic: Text -> SQL
    # We need a schema provider. get_bird_schema should work.
    print("Formatting data...")
    formatted_data = prepare_training_data(bird_examples, get_bird_schema)
    dataset = Dataset.from_list(formatted_data)
    
    # 2. Load Model & Tokenizer
    model_name = config.MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if config.USE_4BIT_QUANTIZATION and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        if device == "cpu":
            model = model.to("cpu")

    model.config.use_cache = False

    # 3. Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16, # Standard rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Training Arguments
    # Optimized for speed for this demo
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1, # Quick demo
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        fp16=(device == "cuda"),
        optim="paged_adamw_32bit" if device == "cuda" else "adamw_torch",
        report_to="none",
        use_cpu=(device == "cpu")
    )
    
    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # 6. Train
    print("Starting training...")
    trainer.train()
    
    # 7. Save
    print("Saving model...")
    trainer.save_model(final_output_dir)
    print("Detailed fine-tuning complete.")

if __name__ == "__main__":
    train_bird()
