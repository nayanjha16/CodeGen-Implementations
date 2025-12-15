"""Inference for query generator: generate MongoDB pipelines from NLQ + predicted schema.

Provides `QueryGenerator` callable and a CLI. The model generates JSON-only
output representing the pipeline (a list of stage dicts). If no trained model
is given, a simple heuristic generator will produce pipelines for basic
aggregation tasks (match/group/project/sort).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_first_json(text: str) -> Optional[Any]:
    # similar to schema predictor helper
    brace_match = re.search(r"\{.*\}|\[.*\]", text, flags=re.S)
    if not brace_match:
        return None
    candidate = brace_match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        # relax by replacing single quotes
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            return None


def build_prompt(nlq: str, pred_schema: Dict[str, List[str]]) -> str:
    lines = [f"{c}: {', '.join(flds)}" for c, flds in pred_schema.items()]
    schema_text = "\n".join(lines)
    prompt = (
        f"NLQ: {nlq}\n"
        "Predicted Schema:\n"
        f"{schema_text}\n"
        "Output (JSON array of pipeline stages only):"
    )
    return prompt


class QueryGenerator:
    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        if model_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            try:
                self.model = PeftModel.from_pretrained(self.model, model_dir)
            except Exception:
                pass
            self.model.to(self.device)

    def predict(self, nlq: str, pred_schema: Dict[str, List[str]], max_new_tokens: int = 256, temperature: float = 0.0) -> List[Dict]:
        if self.model and self.tokenizer:
            prompt = build_prompt(nlq, pred_schema) + " "
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
            decoded = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = extract_first_json(decoded)
            if not parsed or not isinstance(parsed, list):
                logging.warning("Generated output did not contain a JSON list pipeline; falling back to heuristic")
                return self._heuristic(nlq, pred_schema)
            # ensure each stage is a dict
            pipeline = [s for s in parsed if isinstance(s, dict)]
            return pipeline
        else:
            return self._heuristic(nlq, pred_schema)

    def _heuristic(self, nlq: str, pred_schema: Dict[str, List[str]]) -> List[Dict]:
        # Very basic heuristic: if 'count' or 'total' -> $group with $sum; if 'where' like 'country = X' -> $match
        nl = nlq.lower()
        pipeline: List[Dict] = []
        # try to find a field mention
        match = re.search(r"(\w+)\s*(=|is)\s*'?(\w+)'?", nl)
        if match:
            fld, _, val = match.group(1), match.group(2), match.group(3)
            pipeline.append({"$match": {fld: val}})

        if "count" in nl or "number" in nl or "how many" in nl:
            # pick a plausible collection and group by a key
            coll = list(pred_schema.keys())[0] if pred_schema else "collection"
            # choose a numeric field if present
            fields = pred_schema.get(coll, [])
            pipeline.append({"$group": {"_id": None, "count": {"$sum": 1}}})
            pipeline.append({"$project": {"count": 1, "_id": 0}})
            return pipeline

        # default: return a match-only pipeline if we found condition, else empty pipeline
        return pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate Mongo pipeline from NLQ and predicted schema")
    parser.add_argument("--model-dir", help="Trained model dir (optional)")
    parser.add_argument("--nlq", required=True)
    parser.add_argument("--schema-file", required=True, help="JSON file with predicted schema dict")
    args = parser.parse_args()

    with open(args.schema_file, "r", encoding="utf-8") as fh:
        schema = json.load(fh)

    gen = QueryGenerator(model_dir=args.model_dir)
    pipeline = gen.predict(args.nlq, schema)
    print(json.dumps(pipeline, indent=2))


if __name__ == "__main__":
    main()
