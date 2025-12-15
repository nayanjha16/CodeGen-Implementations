"""Inference utilities for schema predictor.

Provides a callable `SchemaPredictor` class which loads a base model + LoRA
adapter and exposes `predict(nlq, schema)` that returns a validated JSON dict
with collections and fields only (no extra text).

If no trained model is provided, a lightweight heuristic fallback is used.
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
    """Extract the first {...} or [...] substring and parse as JSON."""
    # Try to find balanced braces for object
    brace_match = re.search(r"\{.*\}", text, flags=re.S)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    bracket_match = re.search(r"\[.*\]", text, flags=re.S)
    if bracket_match:
        candidate = bracket_match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def build_prompt(nlq: str, schema: Dict[str, List[str]]) -> str:
    schema_lines = []
    for coll, fields in schema.items():
        schema_lines.append(f"{coll}: {', '.join(fields)}")
    schema_text = "\n".join(schema_lines)
    prompt = (
        f"NLQ: {nlq}\n"
        "Schema:\n"
        f"{schema_text}\n"
        "Output (JSON only):"
    )
    return prompt


class SchemaPredictor:
    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        if model_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            # If this is a PEFT adapter, try to load it
            try:
                self.model = PeftModel.from_pretrained(self.model, model_dir)
            except Exception:
                # not a PEFT adapter or already merged
                pass
            self.model.to(self.device)

    def predict(self, nlq: str, schema: Dict[str, List[str]], max_new_tokens: int = 256, temperature: float = 0.0) -> Dict:
        """Return structured JSON dict with collections and fields only.

        If loaded model is not available, use a simple heuristic: return all collections
        whose name or fields are mentioned in the NLQ, otherwise return top-level collections.
        """
        if self.model and self.tokenizer:
            prompt = build_prompt(nlq, schema) + " "
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
            decoded = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # Extract JSON from decoded text
            parsed = extract_first_json(decoded)
            if parsed is None:
                logging.warning("Model output did not contain valid JSON; returning empty result")
                return {"collections": []}
            # Validate against schema
            return self._validate_and_prune(parsed, schema)
        else:
            # heuristic fallback
            return self._heuristic_predict(nlq, schema)

    def _validate_and_prune(self, parsed: Any, schema: Dict[str, List[str]]) -> Dict:
        # Expect parsed to be {"collections": [{"name": ..., "fields": [...]}]}
        if not isinstance(parsed, dict) or "collections" not in parsed:
            logging.warning("Parsed JSON has unexpected format; attempting to coerce.")
            return {"collections": []}

        out = {"collections": []}
        for coll in parsed.get("collections", []):
            name = coll.get("name")
            fields = coll.get("fields", [])
            if name not in schema:
                logging.warning("Predicted collection '%s' not in schema; skipping", name)
                continue
            # Only keep fields that exist in schema for this collection
            valid_fields = [f for f in fields if f in schema.get(name, [])]
            out["collections"].append({"name": name, "fields": valid_fields})
        return out

    def _heuristic_predict(self, nlq: str, schema: Dict[str, List[str]]) -> Dict:
        nlq_low = nlq.lower()
        matches = []
        for coll, fields in schema.items():
            # If collection name appears in NLQ or any field name appears, include it
            if coll.lower() in nlq_low or any((f.lower() in nlq_low) for f in fields):
                # pick fields mentioned in NLQ, else top 3
                selected = [f for f in fields if f.lower() in nlq_low]
                if not selected:
                    selected = fields[:3]
                matches.append({"name": coll, "fields": selected})
        if not matches:
            # default: return top-level collections with empty fields
            matches = [{"name": coll, "fields": []} for coll in list(schema.keys())[:1]]
        return {"collections": matches}


def main():
    parser = argparse.ArgumentParser(description="Run schema predictor inference")
    parser.add_argument("--model-dir", help="Directory with trained model (optional). If omitted, uses heuristic fallback")
    parser.add_argument("--nlq", required=True, help="NLQ text")
    parser.add_argument("--schema-file", required=True, help="Path to JSON file containing schema dict (collection -> fields)")
    args = parser.parse_args()

    with open(args.schema_file, "r", encoding="utf-8") as fh:
        schema = json.load(fh)

    pred = SchemaPredictor(model_dir=args.model_dir)
    out = pred.predict(args.nlq, schema)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
