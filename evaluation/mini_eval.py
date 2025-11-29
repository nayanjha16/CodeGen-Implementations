
from utils.sql_cleaner import normalize_sql

def run_mini_eval(model, retriever, dataset, fewshot):
    correct, total = 0, len(dataset)
    results = []
    for sample in dataset:
        q, schema, gold = sample["question"], sample["schema"], sample["answer_sql"]
        context = retriever.retrieve(q)
        prompt = f"""
You are expert.
{fewshot}
Schema: {schema}
Question: {q}
SQL:
"""
        pred = model.generate_sql(prompt)
        if normalize_sql(pred) == normalize_sql(gold):
            correct += 1
        results.append((q, gold, pred))
    return correct/total, results
