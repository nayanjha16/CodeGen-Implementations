import json
from typing import Dict, Any, List


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_tables(tables_json_path: str) -> Dict[str, dict]:
    """
    Read Spider tables.json and index by db_id.
    """
    tables = load_json(tables_json_path)
    return {t["db_id"]: t for t in tables}
