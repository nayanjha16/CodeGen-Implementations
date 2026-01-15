import sqlglot
from sqlglot import exp

def parse_sql(sql_query: str) -> dict:
    """
    Parses a SQL query into a DB-agnostic QueryPlan.
    """
    try:
        parsed = sqlglot.parse_one(sql_query)
    except Exception as e:
        raise ValueError(f"Failed to parse SQL: {e}")

    # Initialize Query Plan
    query_plan = {
        "collection": None,
        "filters": [],
        "projections": [],
        "group_by": [],
        "aggregates": [],
        "joins": [],
        "sort": [],
        "limit": None
    }

    # 1. Extract Table (Collection)
    # We assume simple queries with one main table for now, or the first table in FROM
    from_expression = parsed.find(exp.From)
    if from_expression:
        for table in from_expression.find_all(exp.Table):
            query_plan["collection"] = table.name
            break # Take the first one as primary
    
    # 2. Extract Filters (WHERE)
    where_expression = parsed.find(exp.Where)
    if where_expression:
        for condition in where_expression.find_all(exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ):
             # This is a simplified extraction. 
             # Complex conditions (OR, nested AND) might need recursion.
             # For now, we extract simple binary comparisons.
             # Check if it's a direct child of the WHERE or an AND
             
             # Extract left (field) and right (value)
             left = condition.left
             right = condition.right
             
             op = ""
             if isinstance(condition, exp.EQ): op = "="
             elif isinstance(condition, exp.GT): op = ">"
             elif isinstance(condition, exp.LT): op = "<"
             elif isinstance(condition, exp.GTE): op = ">="
             elif isinstance(condition, exp.LTE): op = "<="
             elif isinstance(condition, exp.NEQ): op = "!="

             query_plan["filters"].append({
                 "field": left.name,
                 "op": op,
                 "value": _extract_value(right)
             })

    # 3. Extract Projections (SELECT) & Aggregates
    select_expressions = parsed.find_all(exp.Select)
    for select in select_expressions:
        for expression in select.expressions:
            if isinstance(expression, exp.AggFunc):
                 # e.g. COUNT(*), AVG(salary)
                 func_name = expression.sql_name()
                 # Try to find the field inside
                 field_node = expression.find(exp.Column)
                 field_name = field_node.name if field_node else "*"
                 
                 query_plan["aggregates"].append({
                     "func": func_name.lower(),
                     "field": field_name,
                     "alias": expression.alias_or_name
                 })
            elif isinstance(expression, exp.Column):
                query_plan["projections"].append(expression.name)
            elif isinstance(expression, exp.Alias):
                # Handle aliased columns or aggregates
                child = expression.this
                if isinstance(child, exp.AggFunc):
                     func_name = child.sql_name()
                     field_node = child.find(exp.Column)
                     field_name = field_node.name if field_node else "*"
                     query_plan["aggregates"].append({
                         "func": func_name.lower(),
                         "field": field_name,
                         "alias": expression.alias_or_name
                     })
                elif isinstance(child, exp.Column):
                     # Aliased column, currently just storing the column name
                     # If renaming is needed for project, we might need to store alias
                     query_plan["projections"].append(child.name)
            elif isinstance(expression, exp.Star):
                query_plan["projections"].append("*")

    # 4. Extract Group By
    group_expression = parsed.find(exp.Group)
    if group_expression:
        for group_col in group_expression.find_all(exp.Column):
            query_plan["group_by"].append(group_col.name)

    # 5. Extract Joins
    # sqlglot represents joins as expressions in the Select statement usually attached to From?
    # Actually parsed.find_all(exp.Join) works
    for join in parsed.find_all(exp.Join):
        table = join.find(exp.Table)
        table_name = table.name if table else None
        
        # Extract ON condition
        on = join.find(exp.EQ) # Assuming simple equality joins
        left_on = None
        right_on = None
        if on:
            left_on = on.left.name
            right_on = on.right.name
            
        if table_name:
            query_plan["joins"].append({
                "table": table_name,
                "left_on": left_on,
                "right_on": right_on,
                "type": join.kind or "INNER"
            })

    # 6. Extract Order By
    order_expression = parsed.find(exp.Order)
    if order_expression:
        for ordered in order_expression.find_all(exp.Ordered):
            col = ordered.find(exp.Column)
            is_desc = ordered.args.get("desc")
            if col:
                query_plan["sort"].append({
                    "field": col.name,
                    "order": "desc" if is_desc else "asc"
                })

    # 7. Implements Limit
    limit_expression = parsed.find(exp.Limit)
    if limit_expression:
        try:
            query_plan["limit"] = int(limit_expression.expression.this)
        except:
             pass

    # ... (Keep existing simple extraction for compatibility or merge. 
    # For this implementation, we will enhance the existing extraction AND add specific complex detail).
    
    # 8. Extract Having
    having_expression = parsed.find(exp.Having)
    if having_expression:
        # Similar to WHERE
        for condition in having_expression.find_all(exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.NEQ):
             left = condition.left
             right = condition.right
             
             op = ""
             if isinstance(condition, exp.EQ): op = "="
             elif isinstance(condition, exp.GT): op = ">"
             elif isinstance(condition, exp.LT): op = "<"
             elif isinstance(condition, exp.GTE): op = ">="
             elif isinstance(condition, exp.LTE): op = "<="
             elif isinstance(condition, exp.NEQ): op = "!="

             # For HAVING, left side handles aggregates e.g. COUNT(*)
             query_plan.setdefault("having_filters", []).append({
                 "field": left.sql(), # Use raw SQL for aggregate e.g. "count(*)"
                 "op": op,
                 "value": _extract_value(right)
             })

    # 9. Extract Subqueries (in WHERE IN / EXISTS)
    if where_expression:
        for in_exp in where_expression.find_all(exp.In):
            # Check if right side is a subquery
            if isinstance(in_exp.right, exp.Subquery):
                sub_sql = in_exp.right.sql()
                query_plan.setdefault("subqueries", []).append({
                    "type": "IN",
                    "field": in_exp.left.name,
                    "sql": sub_sql
                })
        # Note: sqlglot might represent IN (SELECT ...) differently depending on version
        # We also check for explicit Subquery expression
        
    return query_plan

def is_complex(query_plan: dict) -> bool:
    """Checks if the query plan requires complex mode."""
    if query_plan.get("joins"): return True
    if query_plan.get("having_filters"): return True
    if query_plan.get("subqueries"): return True
    # check for CASE WHEN in aggregates or projections
    # (Simplified: if we see 'case' string in field names/aliases - better to detect in parser)
    return False

def extract_complex_query_plan(sql_query: str) -> dict:
    """
    Wrapper for parse_sql that ensures all complex fields are populated.
    Currently parse_sql does most of it, we just ensure the structure is ready.
    """
    return parse_sql(sql_query)

def _extract_value(node):
    """Helper to extract python value from sqlglot node"""
    if isinstance(node, exp.Literal):
        if node.is_string:
            return node.this
        elif node.is_int:
            return int(node.this)
        elif node.is_number:
            return float(node.this)
    return node.parameters.get("this") or str(node)
