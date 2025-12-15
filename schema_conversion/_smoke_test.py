"""Quick smoke test for extract_schema functionality.

Creates sample SQL files with different constructs and verifies that the
filtering heuristics keep/drop them as expected by calling the module
functions directly.
"""

from pathlib import Path
import tempfile
import json

from schema_conversion.extract_schema import find_sql_and_nl_pairs, is_allowed_sql


def create_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def run_smoke():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # kept case
        kept_sql = td_path / "kept.sql"
        create_file(kept_sql, "SELECT a, count(*) FROM t WHERE a > 1 GROUP BY a HAVING count(*) > 1;")
        create_file(td_path / "kept.nl", "Count rows grouped by a")

        # window function -> drop
        win_sql = td_path / "win.sql"
        create_file(win_sql, "SELECT ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) as rn FROM t;")

        # with/cte -> drop
        with_sql = td_path / "with.sql"
        create_file(with_sql, "WITH cte AS (SELECT 1 as one) SELECT * FROM cte;")

        # subquery -> drop
        sub_sql = td_path / "sub.sql"
        create_file(sub_sql, "SELECT a FROM t WHERE a IN (SELECT b FROM t2);")

        # union -> drop
        union_sql = td_path / "union.sql"
        create_file(union_sql, "SELECT a FROM t UNION SELECT b FROM t2;")

        total = 0
        kept = []
        dropped = []

        for item in find_sql_and_nl_pairs(td):
            total += 1
            allowed, reason = is_allowed_sql(item["sql"])
            if allowed:
                kept.append(item)
            else:
                dropped.append((item, reason))

        print(f"Total: {total}")
        print(f"Kept: {len(kept)}")
        print(f"Dropped: {len(dropped)}")
        print("Dropped reasons:")
        print(json.dumps([r for (_, r) in dropped], indent=2))


if __name__ == "__main__":
    run_smoke()
