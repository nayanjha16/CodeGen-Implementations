def normalize_sql(sql: str):
    sql = sql.strip().rstrip(";").lower()
    sql = " ".join(sql.split())  # collapse whitespace
    return sql
#-------------------


def token_accuracy(predicted, gold):
    p_tokens = set(normalize_sql(predicted).split())
    g_tokens = set(normalize_sql(gold).split())
    return len(p_tokens & g_tokens) / len(g_tokens)