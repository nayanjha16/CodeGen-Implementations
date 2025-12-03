# src/ratsql_t5_lite/schema_graph_text.py

from typing import Dict, List


def _build_table_column_lists(schema: Dict):
    """
    Spider tables.json schema format:
      - "table_names_original": List[str]
      - "column_names_original": List[ [table_idx, col_name], ... ]
      - "column_types": List[str]
      - "primary_keys": List[int]   (column indices)
      - "foreign_keys": List[ [fk_col_idx, pk_col_idx], ... ]
    """
    table_names: List[str] = schema["table_names_original"]
    column_names = schema["column_names_original"]
    column_types = schema["column_types"]
    primary_keys = set(schema.get("primary_keys", []))
    foreign_keys = schema.get("foreign_keys", [])

    # Build per-table columns: {table_idx: [(col_idx, col_name, col_type, is_pk, is_fk)]}
    table_cols = {i: [] for i in range(len(table_names))}
    fk_cols = set([fk for fk, pk in foreign_keys])

    for col_idx, (t_idx, col_name) in enumerate(column_names):
        if t_idx == -1:
            # -1 means "*", pseudo column
            continue
        col_type = column_types[col_idx]
        is_pk = col_idx in primary_keys
        is_fk = col_idx in fk_cols
        table_cols[t_idx].append(
            (col_idx, col_name, col_type, is_pk, is_fk)
        )

    return table_names, table_cols, foreign_keys, primary_keys
        

def schema_to_graph_text(schema: Dict) -> str:
    """
    Turn Spider schema into a text string that exposes tables, columns, types,
    primary keys and foreign keys. This is our "RAT-like" schema graph in text form.
    """
    table_names, table_cols, foreign_keys, primary_keys = _build_table_column_lists(schema)
    parts = []

    # 1) Tables and columns
    for t_idx, t_name in enumerate(table_names):
        cols = table_cols.get(t_idx, [])
        if not cols:
            continue

        col_strs = []
        for col_idx, col_name, col_type, is_pk, is_fk in cols:
            flags = []
            if is_pk:
                flags.append("PK")
            if is_fk:
                flags.append("FK")
            flag_str = ""
            if flags:
                flag_str = "[" + ",".join(flags) + "]"
            col_strs.append(f"{col_name}<{col_type}>{flag_str}")

        table_part = f"[TABLE] {t_name} ( " + " ; ".join(col_strs) + " )"
        parts.append(table_part)

    # 2) Foreign-key relations
    rel_parts = []
    for fk_col, pk_col in foreign_keys:
        # fk_col, pk_col are indices into column_names_original
        # We need table + col name for each
        fk_t_idx, fk_name = schema["column_names_original"][fk_col]
        pk_t_idx, pk_name = schema["column_names_original"][pk_col]
        if fk_t_idx == -1 or pk_t_idx == -1:
            continue
        fk_table = schema["table_names_original"][fk_t_idx]
        pk_table = schema["table_names_original"][pk_t_idx]
        rel_parts.append(
            f"[FK] {fk_table}.{fk_name} -> {pk_table}.{pk_name}"
        )

    if rel_parts:
        parts.append(" | ".join(rel_parts))

    # Final graph text
    graph_text = " ".join(parts)
    return graph_text


def build_ratsql_t5_input(question: str, schema: Dict) -> str:
    """
    Build the input string for RAT-SQL-Lite-T5:
      question + schema graph text
    """
    graph_text = schema_to_graph_text(schema)
    # You can tweak this prompt format, but keep it consistent for train/infer.
    return (
        f"translate question and schema to SQL: "
        f"question: {question} ; schema_graph: {graph_text}"
    )
