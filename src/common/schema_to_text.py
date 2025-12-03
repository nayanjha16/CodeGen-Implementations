def schema_dict_to_text(schema: dict) -> str:
    """
    Convert Spider schema dictionary into a compact string like:
    table1(col1, col2) | table2(colA, colB)
    """
    table_names = schema["table_names_original"]
    columns = schema["column_names_original"]  # list of (table_idx, col_name)

    parts = []
    for t_idx, table_name in enumerate(table_names):
        col_names = [c for (ti, c) in columns if ti == t_idx and c]
        parts.append(f"{table_name}({', '.join(col_names)})")

    return " | ".join(parts)


def build_input_text(question: str, schema: dict) -> str:
    """
    Builds the input string for Text-to-SQL models:
    'translate English to SQL: question: ...; schema: ...'
    """
    schema_text = schema_dict_to_text(schema)
    return f"translate English to SQL: question: {question}; schema: {schema_text}"
