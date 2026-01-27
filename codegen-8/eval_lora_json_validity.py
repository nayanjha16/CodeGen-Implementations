#!/usr/bin/env python3
import argparse, json, random, subprocess, re, time
from typing import Any, Dict, List, Tuple

def run_generate(model: str, adapter_path: str | None, prompt: str, max_tokens: int = 800) -> str:
    cmd = ["mlx_lm.generate", "--model", model, "--prompt", prompt, "--max-tokens", str(max_tokens)]
    if adapter_path:
        cmd += ["--adapter-path", adapter_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = (p.stdout or "").strip()
    if not out:
        out = (p.stderr or "").strip()
    return out

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json(text: str) -> str | None:
    m = JSON_RE.search(text)
    if not m:
        return None
    return m.group(0).strip()

def json_ok(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_object"
    if "collection" not in obj or "operation" not in obj:
        return False, "missing_collection_or_operation"
    op = obj.get("operation")
    if op == "aggregate":
        pipe = obj.get("pipeline")
        if not isinstance(pipe, list) or len(pipe) == 0:
            return False, "bad_pipeline"
    return True, "ok"

def load_samples(path: str, n: int, seed: int = 7) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if not o.get("ok", False):
                continue
            q = (o.get("question") or "").strip()
            sql = (o.get("sql") or "").strip()
            if q and sql:
                items.append(o)
    random.Random(seed).shuffle(items)
    return items[:n]

def prompt_from(item: Dict[str, Any]) -> str:
    q = (item.get("question") or "").strip()
    sql = (item.get("sql") or "").strip()
    return (
        "You are a SQL→MongoDB compiler.\n"
        "Return ONLY a single MongoDB JSON object.\n"
        'Format: {"collection": "...", "operation": "...", ...}\n\n'
        f"Question:\n{q}\n\nSQL:\n{sql}\n"
    )

def eval_model(name: str, model: str, adapter: str | None, samples: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
    ok_json = 0
    ok_schema = 0
    reasons = {}
    t0 = time.time()

    for i, it in enumerate(samples, 1):
        pr = prompt_from(it)
        raw = run_generate(model, adapter, pr, max_tokens=max_tokens)
        js = extract_json(raw)
        if js is None:
            reasons["no_json_found"] = reasons.get("no_json_found", 0) + 1
            continue
        try:
            obj = json.loads(js)
            ok_json += 1
        except Exception:
            reasons["invalid_json"] = reasons.get("invalid_json", 0) + 1
            continue

        good, reason = json_ok(obj)
        if good:
            ok_schema += 1
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

    dt = time.time() - t0
    total = len(samples)
    return {
        "name": name,
        "total": total,
        "valid_json_rate": ok_json / max(1, total),
        "struct_ok_rate": ok_schema / max(1, total),
        "seconds": dt,
        "reasons": reasons
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="nosql_queries_0_2500.repaired.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")
    ap.add_argument("--adapter", default="", help="adapter folder (fused dir ok)")
    ap.add_argument("--max-tokens", type=int, default=900)
    args = ap.parse_args()

    samples = load_samples(args.inp, args.n)
    if not samples:
        raise SystemExit("No ok=true samples found.")

    base = eval_model("BASE", args.model, None, samples, args.max_tokens)
    lora = eval_model("LORA", args.model, args.adapter or None, samples, args.max_tokens)

    print(json.dumps({"base": base, "lora": lora}, indent=2))

if __name__ == "__main__":
    main()

