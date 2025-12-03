import argparse
import os
import json
from datetime import datetime, date

from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
)
import torch


# -------------------------------------------------------------------
# SAFE JSON SERIALIZER (Fixes datetime & unsupported types)
# -------------------------------------------------------------------
def safe_json_dumps(obj):
    """
    Serialize Mongo pipeline list/dict to JSON string.
    Converts datetime/date objects to ISO strings.
    """
    def default(o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return str(o)  # fallback for any other unsupported types

    return json.dumps(obj, default=default)


# -------------------------------------------------------------------
# TOKENIZATION / PREPROCESSING
# -------------------------------------------------------------------
def preprocess(example, tokenizer, max_in_len, max_out_len):
    """
    Tokenization function for SQL → Mongo pairs.
    Converts mongo (list) → JSON string, tokenizes both sides.
    """

    # Tokenize SQL input
    model_inputs = tokenizer(
        example["sql"],
        max_length=max_in_len,
        padding="max_length",
        truncation=True,
    )

    # Convert Mongo pipeline list to JSON string
    mongo_str = safe_json_dumps(example["mongo"])

    # Tokenize Mongo JSON string as labels
    labels = tokenizer(
        mongo_str,
        max_length=max_out_len,
        padding="max_length",
        truncation=True,
    ).input_ids

    model_inputs["labels"] = labels
    return model_inputs


# -------------------------------------------------------------------
# ARGUMENT PARSER
# -------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train T5 for SQL → Mongo translation")

    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to jsonl dataset with fields: sql, mongo",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_output_length", type=int, default=256)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/sql2nosql/checkpoints_t5_sql2mongo",
        help="Where to save trained model",
    )

    return parser.parse_args()


# -------------------------------------------------------------------
# MAIN TRAINING LOGIC
# -------------------------------------------------------------------
def main():
    args = get_args()

    print("\n🔥 Loading dataset from:", args.train_path)
    dataset = load_dataset("json", data_files=args.train_path, split="train")

    print("🔥 Loading tokenizer and T5-base model…")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    # Detect device (CUDA / MPS / CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"🔥 Using device: {device}")
    model.to(device)

    print("🔥 Tokenizing dataset…")
    tokenized_dataset = dataset.map(
    lambda x: preprocess(
        x,
        tokenizer,
        args.max_input_length,
        args.max_output_length
    ),
    batched=False,
    remove_columns=dataset.column_names,   # DROP sql + mongo
    )

    # Training arguments
    training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.lr,
    logging_steps=100,
    save_steps=500,
    save_total_limit=3,
    remove_unused_columns=True,     # ENSURE DICT FIELDS ARE IGNORED
    fp16=torch.cuda.is_available(),
    )



    print("🔥 Initializing Trainer…")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("🔥 Starting training…")
    trainer.train()

    print("🔥 Saving model to:", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n✅ TRAINING COMPLETE!\n")


# -------------------------------------------------------------------
# RUN SCRIPT
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
