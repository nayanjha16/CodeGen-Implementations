def build_lookup_templates(sql_info, schema_graph):
    """
    Build MongoDB $lookup + $unwind templates
    from SQL JOINs and SQLite FK schema.
    RAG v3.1: guarded FK joins only
    """
    lookups = []

    tables = sql_info["tables"]
    root = sql_info["root_table"]

    if not root:
        return []

    for table in tables:
        if table == root:
            continue

        for src_table, relations in schema_graph.items():
            for rel in relations:

                # 🔒 RAG v3.1 GUARD — skip incomplete FK metadata
                if not rel.get("from_column") or not rel.get("to_column"):
                    continue

                # Case 1: table → root
                if src_table == table and rel["to_table"] == root:
                    lookups.append({
                        "$lookup": {
                            "from": table,
                            "localField": rel["to_column"],
                            "foreignField": rel["from_column"],
                            "as": table
                        }
                    })
                    lookups.append({ "$unwind": f"${table}" })

                # Case 2: root → table
                if src_table == root and rel["to_table"] == table:
                    lookups.append({
                        "$lookup": {
                            "from": table,
                            "localField": rel["from_column"],
                            "foreignField": rel["to_column"],
                            "as": table
                        }
                    })
                    lookups.append({ "$unwind": f"${table}" })

    return lookups


def retrieve_rag_context(sql_info, schema_graph):
    """
    RAG v3.x context builder.
    Keeps a stable interface for the pipeline.
    """
    join_templates = build_lookup_templates(sql_info, schema_graph)

    return {
        "tables": ", ".join(sql_info["tables"]),
        "root_collection": sql_info["root_table"],
        "join_templates": join_templates
    }
