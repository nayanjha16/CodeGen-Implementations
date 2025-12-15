"""Extract and filter BIRD SQL + NLQ pairs.

This script scans a BIRD-style input directory (default: `data/bird/`) for
SQL files and JSONL records containing SQL+NLQ pairs, filters them using
sqlparse and a set of rules, and writes the filtered examples as JSON lines
to the output directory (default: `data/bird_filtered/filtered.jsonl`).

Filtering rules (keeps only queries containing only these major constructs):
  - SELECT
  - WHERE
  - JOIN
  - GROUP BY
  - HAVING

Drops queries containing:
  - WINDOW functions / OVER / WINDOW
  - WITH / CTEs / RECURSIVE
  - correlated or any subqueries (detected by `( SELECT ...`)
  - UNION/INTERSECT/EXCEPT
  - ORDER BY / LIMIT / OFFSET (not in allowed set)

Usage:
  python schema_conversion/extract_schema.py --input-dir data/bird --output-dir data/bird_filtered

This script is intentionally conservative (drops queries with subqueries and
other advanced features). It logs how many queries were kept and dropped.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sqlparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def is_allowed_sql(sql: str) -> Tuple[bool, Optional[str]]:
    """Return (allowed, reason) by checking the SQL string for disallowed constructs.

    The check is conservative: any subquery or advanced clause not in the
    allowed set will cause the SQL to be rejected.
    """
    s = sql.strip()
    if not s:
        return False, "empty"

    # Normalize whitespace and lowercase for simple regex checks
    lower = re.sub(r"\s+", " ", s).lower()

    # Ensure it's a SELECT statement
    if not re.match(r"^select\b", lower):
        return False, "not a select statement"

    # Disallowed patterns
    disallowed_patterns = [
        r"\bwith\b",
        r"\bover\b",
        r"\bwindow\b",
        r"\brecursive\b",
        r"\bunion\b",
        r"\bintersect\b",
        r"\bexcept\b",
        r"\border\s+by\b",
        r"\blimit\b",
        r"\boffset\b",
    ]

    for pat in disallowed_patterns:
        if re.search(pat, lower):
            return False, f"contains disallowed pattern: {pat}"

    # Detect subqueries like '( SELECT ...' which we conservatively drop
    if re.search(r"\(\s*select\b", lower):
        return False, "contains subquery"

    # Use sqlparse to validate/parses the statement (catch malformed SQL)
    try:
        parsed = sqlparse.parse(s)
    except Exception as e:  # pragma: no cover - defensive
        return False, f"sqlparse failed: {e}"

    if not parsed:
        return False, "sqlparse produced no statements"

    # If parsing succeded and no disallowed pattern found, allow it.
    return True, None


def find_sql_and_nl_pairs(input_dir: str) -> Iterable[Dict[str, str]]:
    """Yield dicts with keys: db_id, nlq, sql.

    Heuristics:
    - If JSON/JSONL files with 'sql' fields exist, read them.
    - Otherwise, look for .sql files and optional companion .nl / .txt files
      that share the same basename.
    """
    p = Path(input_dir)
    if not p.exists():
        logging.warning("Input directory '%s' does not exist.", input_dir)
        return

    # First, read JSONL/JSON files that contain 'sql'
    for ext in ("*.jsonl", "*.json"):
        for jf in p.rglob(ext):
            try:
                if jf.suffix == ".jsonl":
                    with jf.open("r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            if "sql" in obj:
                                yield {"db_id": obj.get("db_id", ""), "nlq": obj.get("nlq", ""), "sql": obj.get("sql", "")}
                else:  # .json - either one big list or single object
                    with jf.open("r", encoding="utf-8") as fh:
                        obj = json.load(fh)
                    if isinstance(obj, list):
                        for entry in obj:
                            if "sql" in entry:
                                yield {"db_id": entry.get("db_id", ""), "nlq": entry.get("nlq", ""), "sql": entry.get("sql", "")}
                    elif isinstance(obj, dict) and "sql" in obj:
                        yield {"db_id": obj.get("db_id", ""), "nlq": obj.get("nlq", ""), "sql": obj.get("sql", "")}
            except Exception as e:  # pragma: no cover - file read/parsing issues
                logging.warning("Failed to read JSON file %s: %s", jf, e)

    # Next, read .sql files and optional companions
    for sqlf in p.rglob("*.sql"):
        try:
            text = sqlf.read_text(encoding="utf-8")
        except Exception as e:  # pragma: no cover - file read issues
            logging.warning("Failed to read SQL file %s: %s", sqlf, e)
            continue

        # A single file may contain multiple statements; split them
        statements = [s.strip() for s in sqlparse.split(text) if s.strip()]
        nlq = ""
        # Try to find an NLQ companion file
        for ext in (".nl", ".nlq", ".txt", ".question"):
            candidate = sqlf.with_suffix(ext)
            if candidate.exists():
                nlq = candidate.read_text(encoding="utf-8").strip()
                break

        # If multiple statements, yield each separately (db_id as file stem)
        for stmt in statements:
            yield {"db_id": sqlf.stem, "nlq": nlq, "sql": stmt}


def write_jsonl(items: List[Dict[str, str]], out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter BIRD SQL dataset by allowed SQL features.")
    parser.add_argument("--input-dir", default="data/bird", help="Path to BIRD dataset directory")
    parser.add_argument("--output-dir", default="data/bird_filtered", help="Where to write filtered data")
    parser.add_argument("--output-file", default="filtered.jsonl", help="Output JSONL file name")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_file = args.output_file

    kept: List[Dict[str, str]] = []
    dropped_count = 0
    total = 0

    for item in find_sql_and_nl_pairs(input_dir):
        total += 1
        sql = item.get("sql", "")
        allowed, reason = is_allowed_sql(sql)
        if allowed:
            # Normalize whitespace in SQL for output
            item["sql"] = re.sub(r"\s+", " ", sql).strip()
            kept.append({"db_id": item.get("db_id", ""), "nlq": item.get("nlq", ""), "sql": item.get("sql", "")})
        else:
            dropped_count += 1

    out_path = os.path.join(output_dir, output_file)
    write_jsonl(kept, out_path)

    logging.info("Total queries processed: %d", total)
    logging.info("Queries kept: %d", len(kept))
    logging.info("Queries dropped: %d", dropped_count)
    logging.info("Filtered output written to %s", out_path)


if __name__ == "__main__":
    main()
