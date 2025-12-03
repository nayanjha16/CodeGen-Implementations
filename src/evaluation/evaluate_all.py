import argparse
from collections import defaultdict

from src.common.data_utils import load_json, read_tables
from src.evaluation.metrics import exact_match, valid_efficiency_score
from src.evaluation.sql_exec import execution_accuracy, test_suite_accuracy
from src.evaluation.model_wrappers import (
    SQLNetWrapper,
    T5FineTunedWrapper,
    RATSQLT5LiteWrapper,      # NEW CLASS
    GaussAlgoT5Wrapper,
)


def evaluate_models(
    sqlnet_ckpt: str,
    t5_model_dir: str,
    ratsql_t5_dir: str,        # UPDATED
    tables_json: str,
    dev_json: str,
    limit: int = None,
):
    # Load data
    dev = load_json(dev_json)
    tables = read_tables(tables_json)

    if limit is None or limit <= 0 or limit > len(dev):
        limit = len(dev)

    print("\n[Init] Loading models...")
    sqlnet = SQLNetWrapper(sqlnet_ckpt)
    t5_ft = T5FineTunedWrapper(t5_model_dir)
    ratsql_t5 = RATSQLT5LiteWrapper(ratsql_t5_dir)   # UPDATED
    gauss_t5 = GaussAlgoT5Wrapper()

    models = {
        "SQLNet (LSTM Baseline)": sqlnet,
        "T5 Fine-Tuned (Spider)": t5_ft,
        "RAT-SQL-T5-Lite": ratsql_t5,       # UPDATED
        "GaussAlgo T5-Large": gauss_t5,
    }

    stats = {name: defaultdict(float) for name in models.keys()}

    print(f"\n[Eval] Evaluating on {limit} examples from dev_spider.json")

    for i in range(limit):
        ex = dev[i]
        question = ex["question"]
        gold_sql = ex["query"]
        db_id = ex["db_id"]
        schema = tables[db_id]

        for name, model in models.items():
            pred_sql = model.predict(question, schema, db_id)

            em = exact_match(pred_sql, gold_sql)
            ex_acc = execution_accuracy(pred_sql, gold_sql, db_id)
            tsa = test_suite_accuracy(pred_sql, gold_sql, db_id)
            ves = valid_efficiency_score(pred_sql, gold_sql)

            stats[name]["count"] += 1
            stats[name]["em"] += float(em)
            stats[name]["ex"] += float(ex_acc)
            stats[name]["tsa"] += float(tsa)
            stats[name]["ves"] += ves

        if (i + 1) % 50 == 0 or i == limit - 1:
            print(f"  Processed {i + 1}/{limit} examples...")

    print("\n================ Final Evaluation Report ================")
    for name, st in stats.items():
        n = st["count"] or 1.0
        em = st["em"] / n
        ex_acc = st["ex"] / n
        tsa = st["tsa"] / n
        ves = st["ves"] / n

        print(f"\nModel: {name}")
        print(f"  Exact Match Accuracy        : {em:.4f}")
        print(f"  Execution Accuracy          : {ex_acc:.4f}")
        print(f"  Test-Suite Accuracy (approx): {tsa:.4f}")
        print(f"  Valid Efficiency Score      : {ves:.4f}")
    print("\n========================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all Text2SQL models on Spider dev set."
    )
    parser.add_argument("--sqlnet_ckpt", required=True)
    parser.add_argument("--t5_model_dir", required=True)
    parser.add_argument("--ratsql_t5_dir", required=True)   # UPDATED
    parser.add_argument("--tables_json", required=True)
    parser.add_argument("--dev_json", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    evaluate_models(
        sqlnet_ckpt=args.sqlnet_ckpt,
        t5_model_dir=args.t5_model_dir,
        ratsql_t5_dir=args.ratsql_t5_dir,   # UPDATED
        tables_json=args.tables_json,
        dev_json=args.dev_json,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
