import sqlparse
import re
import json
from sqlparse.sql import Identifier, Comparison, Parenthesis, Where, IdentifierList

class UltimateSQLToMongoDB:
    def __init__(self):
        self.ops = {
            '=': '$eq', '>': '$gt', '<': '$lt', 
            '>=': '$gte', '<=': '$lte', '!=': '$ne', 
            '<>': '$ne', 'IN': '$in'
        }

    def convert(self, sql):
        try:
            parsed = sqlparse.parse(sql)[0]
            where_clause = None
            for token in parsed.tokens:
                if isinstance(token, Where):
                    where_clause = self._parse_where_logic(token)
            
            pipeline = [{"$match": where_clause}] if where_clause else []
            return json.dumps(pipeline, indent=2)
        except Exception as e:
            return f"// Parsing Error: {str(e)}"

    def _sql_like_to_regex(self, pattern):
        pattern = re.escape(pattern).replace(r'\%', '.*').replace(r'\_', '.')
        return f"^{pattern}$"

    def _parse_where_logic(self, token_list):
        expressions = []
        current_logic = "$and"
        
        # Clean tokens: no whitespace, no 'WHERE' keyword
        tokens = [t for t in token_list.tokens if not t.is_whitespace and t.value.upper() != 'WHERE']

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # 1. Nesting: ( ... )
            if isinstance(token, Parenthesis):
                expressions.append(self._parse_where_logic(token))

            # 2. Standard Comparisons: x = 1, y > 10, name LIKE 'A%'
            elif isinstance(token, Comparison):
                expressions.append(self._parse_comparison(token))

            # 3. Special Keywords: IS NULL, IN, OR
            elif token.ttype is sqlparse.tokens.Keyword or token.ttype is sqlparse.tokens.Token.Keyword:
                val = token.value.upper()
                
                if val == 'OR': 
                    current_logic = "$or"
                elif val == 'AND': 
                    current_logic = "$and"
                
                # Handling "field IN (1,2,3)" or "field IS NULL"
                elif val in ['IN', 'IS', 'NOT']:
                    # Use a look-back to find the field name
                    field = tokens[i-1].get_real_name() if hasattr(tokens[i-1], 'get_real_name') else tokens[i-1].value
                    
                    # IS NULL / IS NOT NULL
                    if val == 'IS':
                        is_not = (tokens[i+1].value.upper() == 'NOT')
                        target = tokens[i+2] if is_not else tokens[i+1]
                        if target.value.upper() == 'NULL':
                            expressions.append({field: {"$ne": None} if is_not else None})
                            i += (2 if is_not else 1)
                    
                    # IN (val1, val2)
                    elif val == 'IN':
                        list_token = tokens[i+1]
                        # Extract values from parenthesis
                        inner_vals = list_token.value.strip('()').split(',')
                        clean_vals = [v.strip().strip("'\"") for v in inner_vals]
                        # Try to cast to numbers if applicable
                        clean_vals = [int(v) if v.isdigit() else v for v in clean_vals]
                        expressions.append({field: {"$in": clean_vals}})
                        i += 1

            i += 1

        if not expressions: return {}
        return expressions[0] if len(expressions) == 1 else {current_logic: expressions}

    def _parse_comparison(self, comp):
        left = comp.left.get_real_name()
        # Find the operator and right-side value
        op_token = comp.token_next(0, skip_ws=True)[1]
        op = op_token.value.upper()
        right = comp.right.value.strip("'\"")
        
        if right.isdigit(): right = int(right)
        
        if op == 'LIKE':
            return {left: {"$regex": self._sql_like_to_regex(str(right)), "$options": "i"}}
        
        mongo_op = self.ops.get(op, "$eq")
        return {left: right} if mongo_op == "$eq" else {left: {mongo_op: right}}

# --- FINAL TEST ---
converter = UltimateSQLToMongoDB()
test_sql = """
SELECT * FROM orders 
WHERE (status = 'shipped' OR priority > 5) 
AND customer_id IN (101, 102, 105) 
AND tracking_code LIKE 'TRK%' 
AND deleted_at IS NULL
"""
print(converter.convert(test_sql))