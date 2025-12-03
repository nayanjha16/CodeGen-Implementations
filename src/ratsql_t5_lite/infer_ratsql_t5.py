# src/ratsql_t5_lite/infer_ratsql_t5.py

import argparse
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from src.common.data_utils import load_json, read_tables
from src.ratsql_t5_lite.schema_graph_text import build_ratsql_t5_input


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tables_json", required=True)
    parser.add_argument("--dev_json", required=True)
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load tokenizer + model
    tokenizer = T5TokenizerFast.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Load Spider data + schema
    dev = load_json(args.dev_json)
    tables = read_tables(args.tables_json)

    ex = dev[args.idx]
    question = ex["question"]
    gold_sql = ex["query"]
    db_id = ex["db_id"]
    schema = tables[db_id]

    # Build RAT-SQL-T5 formatted input
    input_text = build_ratsql_t5_input(question, schema)

    print("\n=== RAT-SQL-Lite-T5 Inference ===")
    print("DB:", db_id)
    print("Question:", question)
    print("[ GOLD SQL ]:", gold_sql)
    print("\n[ Input text ]:", input_text)

    # Tokenize input
    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=768,   # avoid truncation!
    ).to(device)

    # Decode SQL
    with torch.no_grad():
        gen_ids = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_beams=4,
            max_length=160,
            early_stopping=True,
            num_return_sequences=1,
        )

    pred_sql = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print("\n[PREDICTED SQL]:", pred_sql, "\n")


if __name__ == "__main__":
    main()
