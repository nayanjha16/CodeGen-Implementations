class SQLGenerator:
    def __init__(self, llm_client, nosql_target="mongodb", sql_model="phi3-finetuned", nosql_model="qwen2.5-coder:3b"):
        self.llm = llm_client
        self.nosql_target = nosql_target
        self.sql_model = sql_model
        self.nosql_model = nosql_model

    def generate_sql(self, question, subgraph):
        schema = "\n".join(d['text'] for _, d in subgraph.nodes(data=True))
        prompt = (
            "Generate only sql query. No explanation.\n"
            f"Schema:\n{schema}\n"
            f"Question:{question}\n"
            "SQL:"
        )
        return self.llm.generate(prompt, model=self.sql_model)

    def generate_nosql(self, sql_query, subgraph=None, target=None):
        target = (target or self.nosql_target).strip().lower()
        
        # Extract field names and schema details from subgraph
        schema_text = ""
        field_names = set()
        if subgraph:
            schema_lines = []
            for node_id, d in subgraph.nodes(data=True):
                text = d.get('text', '')
                if text:
                    schema_lines.append(text)
                # Extract field names (columns)
                if d.get('kind') == 'column':
                    col = d.get('column', '')
                    if col:
                        field_names.add(col.lower())
            schema_text = "\n".join(schema_lines)
        
        # Build field reference list
        field_list = "Available fields: " + ", ".join(sorted(field_names)) if field_names else ""
        
        prompt = (
            f"Translate the following SQL query into an equivalent {target} query/pipeline.\n"
            f"IMPORTANT: Use EXACT lowercase field names from the schema. Do NOT capitalize field names.\n"
            f"{field_list}\n\n"
            f"Requirements:\n"
            f"1. Use lowercase field names (never capitalize them)\n"
            f"2. Always include $project to select only requested columns\n"
            f"3. Preserve sort order (ASC/DESC) from SQL\n"
            f"4. Return ONLY the {target} code, no explanation.\n\n"
            f"Schema:\n{schema_text}\n\n"
            f"SQL:\n{sql_query}\n\n"
            f"{target.upper()}:"
        )
        return self.llm.generate(prompt, model=self.nosql_model)
