import argparse
import torch
from transformers import T5TokenizerFast

from src.common.data_utils import read_tables
from src.common.schema_to_text import build_input_text
from src.sqlnet.model import SQLNet


def infer_custom(ckpt, tables_json, question, db_id):
    # Load schema
    tables = read_tables(tables_json)
    schema = tables[db_id]

    # Build input
    input_text = build_input_text(question, schema)

    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    model = SQLNet.load_from_checkpoint(ckpt, tokenizer=tokenizer)
    model.eval()

    # Tokenize
    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        output_ids = model.generate(enc["input_ids"], max_length=100)[0]

    pred_sql = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("\n===============================")
    print("CUSTOM SQL INFERENCE")
    print("===============================")
    print(f"Question: {question}")
    print(f"DB: {db_id}")
    print(f"Predicted SQL: {pred_sql}")
    print("===============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tables_json", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--db_id", required=True)

    args = parser.parse_args()

    infer_custom(
        args.ckpt,
        args.tables_json,
        args.question,
        args.db_id,
    )
