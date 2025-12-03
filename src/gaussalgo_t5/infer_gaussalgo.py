# src/gaussalgo_t5/infer_gaussalgo.py

import argparse
from src.gaussalgo_t5.model import GaussAlgoT5Text2SQL
from src.common.data_utils import load_json, read_tables
from src.common.schema_to_text import build_input_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_json", required=True, help="Path to Spider tables.json")
    parser.add_argument("--dev_json", required=True, help="Path to Spider dev_spider.json")
    parser.add_argument("--idx", type=int, default=0, help="Index into dev_spider.json")
    args = parser.parse_args()

    # Load Spider dev + tables
    dev_data = load_json(args.dev_json)
    tables = read_tables(args.tables_json)

    ex = dev_data[args.idx]
    question = ex["question"]
    db_id = ex["db_id"]
    gold_sql = ex["query"]
    schema = tables[db_id]

    # Same input format you used for SQLNet and T5:
    # "translate English to SQL: question: ...; schema: table1(... ) | ..."
    input_text = build_input_text(question, schema)

    print("\n[GaussAlgo T5-LM-Large Text2SQL]")
    print("DB ID   :", db_id)
    print("Index   :", args.idx)
    print("Question:", question)
    print("\n[Input Text]:")
    print(input_text)

    # Load model 4
    model = GaussAlgoT5Text2SQL()

    # Generate SQL
    pred_sql = model.generate(input_text)

    print("\n[Gold SQL]:")
    print(gold_sql)

    print("\n[Predicted SQL (GaussAlgo T5)]:")
    print(pred_sql)
    print()


if __name__ == "__main__":
    main()
