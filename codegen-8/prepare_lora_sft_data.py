#!/usr/bin/env python3
# Build a tiny LoRA SFT dataset (chat-style JSONL) from your repaired outputs.
# Input:  nosql_queries_0_2500.repaired.jsonl
# Output: lora_data/train.jsonl + lora_data/valid.jsonl

import argparse, json, os, random
from pathlib import Path

SYSTEM = (
  "You are a SQL→MongoDB compiler.\n"
  "Given a natural language question and its SQL, output ONLY a MongoDB JSON object with:\n"
  '{ "collection": <string>, "operation": "aggregate"|"find"|..., "pipeline": [...] or "filter": {...} }\n'
  "Rules:\n"
  "- Output valid JSON only (no markdown, no commentary).\n"
  "- Prefer aggregation pipelines when grouping/sorting.\n"
  "- Do not invent collections or fields.\n"
)

def norm_nosql(nosql_value):
    # Your file sometimes stores nosql as dict/string.
    if nosql_value is None:
        return None
    if isinstance(nosql_value, dict):
        return json.dumps(nosql_value, ensure_ascii=False)
    if isinstance(nosql_value, str):
        s = nosql_value.strip()
        # ensure it is JSON-ish; if it's already JSON string, keep it
        return s
    return json.dumps(nosql_value, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="nosql_queries_0_2500.repaired.jsonl")
    ap.add_argument("--out", default="lora_data", help="output dir")
    ap.add_argument("--n", type=int, default=20, help="use first N ok=true samples")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--valid", type=int, default=2, help="validation count")
    args = ap.parse_args()

    random.seed(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if not o.get("ok", False):
                continue
            q = (o.get("question") or "").strip()
            sql = (o.get("sql") or "").strip()
            nosql = norm_nosql(o.get("nosql"))
            if not q or not sql or not nosql:
                continue

            user = (
                "Convert this SQL result computation into an equivalent MongoDB query JSON.\n\n"
                f"Question:\n{q}\n\n"
                f"SQL:\n{sql}\n\n"
                "Return ONLY JSON."
            )

            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": nosql},
                ]
            })

            if len(samples) >= args.n:
                break

    if len(samples) < max(3, args.valid + 1):
        raise SystemExit(f"Not enough samples collected: {len(samples)}")

    # simple split: last `valid` as valid
    train = samples[:-args.valid]
    valid = samples[-args.valid:]

    with open(outdir / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(outdir / "valid.jsonl", "w", encoding="utf-8") as f:
        for s in valid:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"wrote: {outdir/'train.jsonl'} ({len(train)})")
    print(f"wrote: {outdir/'valid.jsonl'} ({len(valid)})")

if __name__ == "__main__":
    main()

