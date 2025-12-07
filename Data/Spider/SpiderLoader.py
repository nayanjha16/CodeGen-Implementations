import json
import os
import sqlparse


class SpiderLoader:
    """
    Loads Spider dataset entries and parses either schema.json or schema.sql.
    """

    def __init__(self, root):
        self.root = root
        self.db_dir = os.path.join(root, "database")

    def _load_db_schema(self, db_id):
        json_path = os.path.join(self.db_dir, db_id, "schema.json")
        sql_path = os.path.join(self.db_dir, db_id, "schema.sql")

        # If JSON exists, use it
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            schema["db_id"] = db_id
            return schema

        # Otherwise: parse schema.sql
        if not os.path.exists(sql_path):
            raise FileNotFoundError(f"No schema.json or schema.sql found for {db_id}")

        return self._parse_sql_schema(sql_path, db_id)

    def _parse_sql_schema(self, sql_path, db_id):
        """
        Convert Spider schema.sql into schema.json format.
        """
        with open(sql_path, "r", encoding="utf-8") as f:
            sql_text = f.read()

        parsed = sqlparse.parse(sql_text)

        table_names = []
        column_names = []
        column_types = []
        primary_keys = []
        foreign_keys = []

        table_idx_map = {}
        col_global_idx = 0

        for stmt in parsed:
            if stmt.get_type() != "CREATE":
                continue

            tokens = [t for t in stmt.tokens if not t.is_whitespace]
            table_name = None
            columns = []

            # Detect CREATE TABLE
            for i, t in enumerate(tokens):
                if t.match(sqlparse.tokens.Keyword, "TABLE"):
                    table_name = tokens[i + 2].value.strip("`\"")
                    break

            if not table_name:
                continue

            table_idx_map[table_name] = len(table_names)
            table_names.append(table_name)

            # Extract columns block
            parenthesis = [t for t in tokens if t.ttype is None and "(" in str(t)]
            if not parenthesis:
                continue

            col_def_block = parenthesis[0].value.strip("()")
            lines = [l.strip() for l in col_def_block.split(",")]

            for line in lines:
                parts = line.split()

                # PRIMARY KEY declaration (table-level)
                if parts[0].upper() == "PRIMARY":
                    # PRIMARY KEY(col1, col2)
                    cols = line[line.find("(")+1 : line.find(")")]
                    pk_cols = [c.strip("` ") for c in cols.split(",")]
                    # map to indices later
                    continue

                col_name = parts[0].strip("`")
                col_type = parts[1].upper() if len(parts) > 1 else "TEXT"

                column_names.append([table_idx_map[table_name], col_name])
                column_types.append(col_type)

                # Check inline PK
                if "PRIMARY" in line.upper():
                    primary_keys.append(col_global_idx)

                # Check FK
                if "REFERENCES" in line.upper():
                    try:
                        ref_parts = line.split("REFERENCES")
                        if len(ref_parts) < 2:
                            col_global_idx += 1
                            continue
                        
                        ref_table = ref_parts[1].split("(")[0].strip(" `")
                        ref_col = (
                            line.split("(")[2].split(")")[0].strip(" `")
                            if line.count("(") > 1
                            else line.split("(")[1].split(")")[0].strip(" `")
                        )

                        # Find referenced col index
                        ref_table_idx = table_idx_map.get(ref_table)
                        if ref_table_idx is not None:
                            # find column index
                            for idx, (t_idx, cname) in enumerate(column_names):
                                if t_idx == ref_table_idx and cname == ref_col:
                                    foreign_keys.append([col_global_idx, idx])
                                    break
                    except (IndexError, ValueError):
                        # Skip malformed FK declarations
                        pass

                col_global_idx += 1

        return {
            "db_id": db_id,
            "table_names": table_names,
            "column_names": column_names,
            "column_types": column_types,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
        }

    def load_examples(self, split="dev"):
        path = os.path.join(self.root, f"{split}.json")
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        skipped_dbs = set()  # Track which db_ids have been skipped

        for ex in examples:
            # Some entries don't have schemas available in Spider, skip for now.
            try:
                ex["db_schema"] = self._load_db_schema(ex["db_id"])
            except FileNotFoundError:
                # Only log once per db_id
                if ex["db_id"] not in skipped_dbs:
                    print(f"Skipping {ex['db_id']} due to missing schema file.")
                    skipped_dbs.add(ex["db_id"])
                continue
            yield ex
