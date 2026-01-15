"""
Fine-tuning script for Text-to-SQL using QLoRA.
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
from dataset_loader import load_spider_dataset, load_bird_dataset, load_ddl_dataset, get_unified_schema
from dataset_formatter import prepare_training_data

def train(limit=None, force=False):
    print(f"Starting fine-tuning for {config.MODEL_NAME}...")
    
    # 0. Check for existing checkpoint
    final_output_dir = os.path.join(config.TRAINING_ARGS["output_dir"], "final_checkpoint")
    if os.path.exists(final_output_dir) and not force:
        print(f"✓ Checkpoint already exists at {final_output_dir}")
        print("Skipping training. Use force=True to overwrite.")
        return
    
    # 1. Load Datasets
    # Spider: 4245-5660
    print("Loading Spider dataset (slice: 4245-5660)...")
    spider_examples = load_spider_dataset(split='train', slice_range=(4245, 5660), limit=limit)
    
    # BirdBench: 4783-6376
    print("Loading BirdBench dataset (slice: 4783-6376)...")
    bird_examples = load_bird_dataset(split='train', slice_range=(4783, 6376), limit=limit)
    
    # DDL/DML: Synthetic
    print("Loading DDL/DML dataset...")
    ddl_examples = load_ddl_dataset(limit=limit)
    
    combined_examples = spider_examples + bird_examples + ddl_examples
    print(f"Combined: {len(spider_examples)} Spider + {len(bird_examples)} Bird + {len(ddl_examples)} DDL = {len(combined_examples)} total examples.")
    
    # Format data using unified schema loader
    print("Formatting data...")
    formatted_data = prepare_training_data(combined_examples, get_unified_schema)
    dataset = Dataset.from_list(formatted_data)
    print(f"Formatted {len(dataset)} examples for training.")
    
    # 2. Load Model & Tokenizer
    print("Loading model...")
    
    if config.USE_4BIT_QUANTIZATION:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager" 
        )
        model = prepare_model_for_kbit_training(model)
        # Force config to reflect float16 to avoid internal BF16 assumptions
        model.config.torch_dtype = torch.float16
    else:
        print(f"Loading model on {config.DEVICE} (no quantization)...")
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            device_map="auto" if config.DEVICE == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
        if config.DEVICE == "cpu":
            model = model.to("cpu")
            

            
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 3. Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=config.LORA_CONFIG["r"],
        lora_alpha=config.LORA_CONFIG["lora_alpha"],
        lora_dropout=config.LORA_CONFIG["lora_dropout"],
        bias=config.LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.LORA_CONFIG["target_modules"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # CRITICAL: Force model to float16 to prevent BFloat16 mix-ups on T4
    # We do this AFTER get_peft_model so that the new LoRA adapters are also covered.
    if config.DEVICE == "cuda":
        print("Checking for BFloat16 parameters to convert...")
        for name, module in model.named_modules():
            # Convert parameters
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype == torch.bfloat16:
                     print(f"Casting param {name}.{param_name} to float16")
                     param.data = param.data.to(torch.float16)
            # Convert buffers (e.g. batch norm stats)
            for buf_name, buf in module.named_buffers(recurse=False):
                if buf.dtype == torch.bfloat16:
                     print(f"Casting buffer {name}.{buf_name} to float16")
                     buf.data = buf.data.to(torch.float16)
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=config.TRAINING_ARGS["output_dir"],
        num_train_epochs=config.TRAINING_ARGS["num_train_epochs"],
        max_steps=config.TRAINING_ARGS.get("max_steps", -1),
        per_device_train_batch_size=config.TRAINING_ARGS["per_device_train_batch_size"],
        gradient_accumulation_steps=config.TRAINING_ARGS["gradient_accumulation_steps"],
        learning_rate=config.TRAINING_ARGS["learning_rate"],
        logging_steps=config.TRAINING_ARGS["logging_steps"],
        save_steps=config.TRAINING_ARGS["save_steps"],
        fp16=config.TRAINING_ARGS["fp16"],
        bf16=False, # Explicitly disable BFloat16
        optim=config.TRAINING_ARGS["optim"],
        report_to="none",
        push_to_hub=False,
        use_cpu=True if config.DEVICE == "cpu" else False,
    )
    
    # Debug: Check model device
    print(f"Model device: {model.device}")
    first_param = next(model.parameters())
    print(f"First parameter device: {first_param.device}")
    
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
    trainer.save_model(os.path.join(config.TRAINING_ARGS["output_dir"], "final_checkpoint"))
    print("Done!")

if __name__ == "__main__":
    train()
