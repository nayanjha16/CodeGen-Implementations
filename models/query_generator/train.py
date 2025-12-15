"""Train a model to generate MongoDB aggregation pipelines from NLQ + predicted schema.

This mirrors the schema predictor trainer but targets pipeline JSON as the
supervised target. Training expects a JSONL file with fields: `nlq`, `pred_schema`
(a dict mapping collection->fields) and `gold_pipeline` (the target pipeline, as
JSON serializable list). If `gold_pipeline` is missing the example is skipped.

The training routine uses a causal LM with LoRA (PEFT) and the HF Trainer.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_prompt(nlq: str, pred_schema: Dict[str, List[str]]) -> str:
    schema_lines = []
    for coll, fields in pred_schema.items():
        schema_lines.append(f"{coll}: {', '.join(fields)}")
    schema_text = "\n".join(schema_lines)
    prompt = (
        "Instruction: Given the natural language question and predicted DB schema (collections and fields), "
        "generate a MongoDB aggregation pipeline (JSON array of stages).\n\n"
        f"NLQ: {nlq}\n\n"
        "Predicted Schema:\n"
        f"{schema_text}\n\n"
        "Output (JSON only):"
    )
    return prompt


def prepare_examples(path: str) -> List[Dict]:
    ex = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            nlq = obj.get("nlq")
            pred_schema = obj.get("pred_schema")
            gold = obj.get("gold_pipeline")
            if not (nlq and pred_schema and gold):
                continue
            prompt = build_prompt(nlq, pred_schema)
            target = json.dumps(gold, ensure_ascii=False)
            ex.append({"prompt": prompt, "target": target})
    return ex


def tokenize_and_build_labels(examples: List[Dict], tokenizer, max_length: int = 1024):
    inputs = []
    for e in examples:
        prompt = e["prompt"]
        target = e["target"]
        full = prompt + "\n" + target + tokenizer.eos_token
        tokenized = tokenizer(full, truncation=True, max_length=max_length)
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])
        labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
        labels = labels[: len(tokenized["input_ids"])]
        inputs.append({"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"], "labels": labels})
    return Dataset.from_list(inputs)


def main():
    parser = argparse.ArgumentParser(description="Train query generator model with LoRA")
    parser.add_argument("--train-file", required=True, help="JSONL containing nlq, pred_schema, gold_pipeline")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    examples = prepare_examples(args.train_file)
    if not examples:
        logging.error("No training examples found in %s", args.train_file)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = tokenize_and_build_labels(examples, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj", "v_proj"], inference_mode=False)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(output_dir=args.output_dir, per_device_train_batch_size=args.per_device_batch_size, num_train_epochs=args.epochs, logging_steps=10, fp16=torch.cuda.is_available(), save_total_limit=3, report_to=[])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=data_collator)
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info("Training finished, model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
