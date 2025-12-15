"""Train a schema prediction model (NLQ + DB schema -> collections + fields).

This script prepares training examples from an input JSONL file where each
line contains at least `nlq`, `db_id`, and `schema` (schema is a dict mapping
collection name to list of field names). It fine-tunes a causal LM using LoRA
via PEFT and saves checkpoints to `--output-dir`.

Notes:
 - The trainer is implemented with the Hugging Face `Trainer` for simplicity.
 - Training expects the `transformers`, `peft`, and `datasets` packages.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_prompt(nlq: str, schema: Dict[str, List[str]]) -> str:
    """Create a text prompt from NLQ and schema.

    Schema is formatted as lines: <collection>: field1, field2, ...
    The model should generate structured JSON only (one JSON object) as response.
    """
    schema_lines = []
    for coll, fields in schema.items():
        schema_lines.append(f"{coll}: {', '.join(fields)}")
    schema_text = "\n".join(schema_lines)

    prompt = (
        "Instruction: Given the natural language question and the database schema, "
        "predict which MongoDB collections and their fields are relevant.\n\n"
        f"NLQ: {nlq}\n\n"
        "Schema:\n"
        f"{schema_text}\n\n"
        "Output (JSON only): {\"collections\": [ {\"name\": <collection>, \"fields\": [<field>, ...] }, ... ] }"
    )
    return prompt


def build_target(schema_subset: Dict[str, List[str]]) -> str:
    """Serialize target JSON (collections + fields)."""
    return json.dumps({"collections": [{"name": k, "fields": v} for k, v in schema_subset.items()]}, ensure_ascii=False)


def prepare_examples_from_jsonl(path: str) -> List[Dict]:
    """Expect each line to contain 'nlq', 'db_id', and 'schema' (map of coll->fields).

    For supervised training we need a target; in many cases the gold mapping might
    be provided as `gold_schema` in the input lines; otherwise we default to using
    the full schema as target (this is convenient for starting training).
    """
    examples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            nlq = obj.get("nlq", "")
            schema = obj.get("schema") or obj.get("db_schema") or obj.get("mongo_schema")
            if not schema:
                # skip examples without schema
                continue
            gold = obj.get("gold_schema") or schema

            prompt = build_prompt(nlq, schema)
            target = build_target(gold)
            examples.append({"prompt": prompt, "target": target})
    return examples


def tokenize_and_build_labels(examples: List[Dict], tokenizer, max_length: int = 1024) -> Dataset:
    # Build sequences where input is prompt and labels are target only.
    inputs = []
    for ex in examples:
        prompt = ex["prompt"]
        target = ex["target"]
        # Join so model sees prompt then target (we train causal LM to predict target tokens)
        full = prompt + "\n" + target + tokenizer.eos_token
        tokenized = tokenizer(full, truncation=True, max_length=max_length)
        input_ids = tokenized["input_ids"]
        # tokenized prompt length to mask labels
        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        # Ensure labels same length as input_ids
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))
        if len(labels) > len(input_ids):
            labels = labels[: len(input_ids)]

        inputs.append({"input_ids": input_ids, "attention_mask": tokenized["attention_mask"], "labels": labels})

    ds = Dataset.from_list(inputs)
    return ds


def main():
    parser = argparse.ArgumentParser(description="Train schema predictor with LoRA")
    parser.add_argument("--train-file", required=True, help="Input JSONL with training examples (nlq, schema, optional gold_schema)")
    parser.add_argument("--model-name", required=True, help="Base model name or path (1–2B recommended)")
    parser.add_argument("--output-dir", required=True, help="Where to save model checkpoints")
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--per-device-batch-size", default=4, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--lora-r", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16, type=int)
    parser.add_argument("--max-length", default=1024, type=int)
    args = parser.parse_args()

    examples = prepare_examples_from_jsonl(args.train_file)
    if not examples:
        logging.error("No training examples found in %s", args.train_file)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = tokenize_and_build_labels(examples, tokenizer, max_length=args.max_length)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    # Apply LoRA via PEFT
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj", "v_proj"], inference_mode=False)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=[],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator)

    trainer.train()

    # Save PEFT adapter and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info("Training complete. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
