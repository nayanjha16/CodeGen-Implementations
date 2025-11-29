
import re

def extract_last_sql(decoded: str) -> str:
    text = decoded.replace("```sql", "").replace("```", "")
    parts = text.split("SQL:")
    candidate = parts[-1].strip()
    for stop in ["###", "Schema:", "Question:"]:
        idx = candidate.find(stop)
        if idx != -1:
            candidate = candidate[:idx]
    return candidate.strip()

def normalize_sql(sql: str) -> str:
    sql = sql.lower().replace(";", "")
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip()
