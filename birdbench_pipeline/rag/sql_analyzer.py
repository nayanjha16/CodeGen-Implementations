import sqlglot
from sqlglot.expressions import Table, Join, EQ

def analyze_sql(sql: str):
    parsed = sqlglot.parse_one(sql)

    tables = []
    joins = []

    for table in parsed.find_all(Table):
        if table.name not in tables:
            tables.append(table.name)

    for join in parsed.find_all(Join):
        on_expr = join.args.get("on")
        if isinstance(on_expr, EQ):
            left = on_expr.left
            right = on_expr.right
            joins.append({
                "left": left.sql(),
                "right": right.sql()
            })

    root_table = tables[0] if tables else None

    return {
        "tables": tables,
        "root_table": root_table,
        "joins": joins
    }
