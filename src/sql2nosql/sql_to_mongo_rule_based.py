"""
Model 1: Rule-Based SQL → MongoDB Aggregation Pipeline Converter
Supports:
  - SELECT
  - FROM
  - JOIN → $lookup
  - WHERE → $match (AND/OR + comparisons)
  - ORDER BY → $sort
  - LIMIT → $limit
"""

from typing import Any, Dict, List, Optional
from pprint import pprint
import sqlglot
from sqlglot import exp


class SQLToMongoConverter:

    OPERATOR_MAP = {
        "eq": "$eq",
        "neq": "$ne",
        "gt": "$gt",
        "lt": "$lt",
        "gte": "$gte",
        "lte": "$lte",
    }

    def convert(self, sql: str) -> List[Dict[str, Any]]:
        """
        Main entry point. Convert SQL → MongoDB pipeline.
        """
        try:
            parsed = sqlglot.parse_one(sql)
        except Exception as e:
            raise ValueError(f"SQL parse error: {e}")

        if not isinstance(parsed, exp.Select):
            raise NotImplementedError("Only SELECT statements are supported.")

        self.parsed = parsed  # store AST

        pipeline = []

        # JOINS → $lookup
        pipeline.extend(self._build_lookups(parsed))

        # WHERE → $match
        where_expr = parsed.args.get("where")
        match_stage = self._build_match_stage(where_expr)
        if match_stage:
            pipeline.append(match_stage)

        # SELECT → $project
        proj = self._build_projection(parsed)
        if proj:
            pipeline.append(proj)

        # ORDER BY → $sort
        sort_stage = self._build_sort(parsed)
        if sort_stage:
            pipeline.append(sort_stage)

        # LIMIT → $limit
        limit_stage = self._build_limit(parsed)
        if limit_stage:
            pipeline.append(limit_stage)

        return pipeline

    # ----------------------------------------------------------------------
    # FROM / Main Table Resolution
    # ----------------------------------------------------------------------
    def _get_main_table(self, select: exp.Select) -> Optional[str]:
        """
        Get the main table from FROM clause.
        Handles:
            FROM A
            FROM A AS X
            FROM A JOIN B ...
        """
        from_expr = select.args.get("from") or select.args.get("from_")
        if not isinstance(from_expr, exp.From):
            return None

        # FROM A
        if isinstance(from_expr.this, exp.Table):
            return from_expr.this.name

        # FROM A AS X
        if isinstance(from_expr.this, exp.Alias) and isinstance(from_expr.this.this, exp.Table):
            return from_expr.this.this.name

        # Fallback: scan for tables
        for node in from_expr.walk():
            if isinstance(node, exp.Table):
                return node.name

        return None

    # ----------------------------------------------------------------------
    # JOIN → $lookup
    # ----------------------------------------------------------------------
    def _build_lookups(self, select: exp.Select) -> List[Dict[str, Any]]:
        lookups = []

        joins = select.args.get("joins") or select.joins
        if not joins:
            return lookups

        main_table = self._get_main_table(select)

        for join in joins:
            if not isinstance(join, exp.Join):
                continue

            # joined table (JOIN B)
            table_expr = join.this
            if isinstance(table_expr, exp.Table):
                joined_table = table_expr.name
                joined_alias = table_expr.alias_or_name
            elif isinstance(table_expr, exp.Alias) and isinstance(table_expr.this, exp.Table):
                joined_table = table_expr.this.name
                joined_alias = table_expr.alias
            else:
                continue

            # ON A.col = B.col
            on = join.args.get("on")
            if not on or not isinstance(on, exp.EQ):
                continue

            left = self._extract_col_ref(on.left)
            right = self._extract_col_ref(on.right)

            if not left or not right:
                continue

            # Determine main vs joined table side
            if left["table"] == main_table:
                local_field = left["column"]
                foreign_field = right["column"]
            else:
                local_field = right["column"]
                foreign_field = left["column"]

            lookup_stage = {
                "$lookup": {
                    "from": joined_table,
                    "localField": local_field,
                    "foreignField": foreign_field,
                    "as": joined_alias,
                }
            }

            lookups.append(lookup_stage)

        return lookups

    # Extract table + column
    def _extract_col_ref(self, expr):
        if isinstance(expr, exp.Column):
            table = expr.table
            col = expr.name
            return {"table": table, "column": col}
        return None

    # ----------------------------------------------------------------------
    # WHERE → $match
    # ----------------------------------------------------------------------
    def _build_match_stage(self, where_expr):
        if not where_expr:
            return None

        mongo_filter = self._convert_condition(where_expr.this)
        if mongo_filter:
            return {"$match": mongo_filter}

        return None

    def _convert_condition(self, expr):
        """
        Convert SQLGlot WHERE tree into MongoDB expressions.
        Supports:
            AND / OR
            =, !=, >, <, >=, <=
        """

        # AND
        if isinstance(expr, exp.And):
            return {
                "$and": [
                    self._convert_condition(expr.left),
                    self._convert_condition(expr.right),
                ]
            }

        # OR
        if isinstance(expr, exp.Or):
            return {
                "$or": [
                    self._convert_condition(expr.left),
                    self._convert_condition(expr.right),
                ]
            }

        # Comparisons
        if isinstance(expr, (exp.EQ, exp.NEQ, exp.GT, exp.LT, exp.GTE, exp.LTE)):
            left = self._extract_col_ref(expr.left)
            right_val = expr.right

            op_name = expr.key.lower()
            mongo_op = self.OPERATOR_MAP.get(op_name)

            # literal value extraction
            if isinstance(right_val, exp.Literal):
                value = right_val.this
            else:
                value = str(right_val)

            # table.column format if needed
            if left["table"]:
                col_name = f"{left['table']}.{left['column']}"
            else:
                col_name = left["column"]

            return {col_name: {mongo_op: value}}

        return {}

    # ----------------------------------------------------------------------
    # SELECT → $project
    # ----------------------------------------------------------------------
    def _build_projection(self, select):
        """
        Only include selected columns.
        """
        exprs = select.expressions
        if not exprs:
            return None

        proj = {"_id": 0}

        for col in exprs:
            if isinstance(col, exp.Column):
                proj[col.name] = 1

        return {"$project": proj}

    # ----------------------------------------------------------------------
    # ORDER BY → $sort
    # ----------------------------------------------------------------------
    def _build_sort(self, select):
        order = select.args.get("order")
        if not isinstance(order, exp.Order):
            return None

        sort_spec = {}

        for item in order.expressions:
            col = item.this
            direction = -1 if item.args.get("desc") else 1

            if isinstance(col, exp.Column):
                sort_spec[col.name] = direction

        return {"$sort": sort_spec} if sort_spec else None

    # ----------------------------------------------------------------------
    # LIMIT → $limit
    # ----------------------------------------------------------------------
    def _build_limit(self, select):
        limit = select.args.get("limit")
        if not isinstance(limit, exp.Limit):
            return None

        val = limit.args.get("expression")
        if isinstance(val, exp.Literal):
            return {"$limit": int(val.this)}

        return None


# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sql = """
    SELECT A.name, B.title
    FROM A
    JOIN B ON A.id = B.a_id
    WHERE A.age > 30 AND B.city != 'Delhi'
    ORDER BY A.name ASC, B.title DESC
    LIMIT 10;
    """

    converter = SQLToMongoConverter()
    pprint(converter.convert(sql))
