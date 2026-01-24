from sqlalchemy import create_engine, inspect
import json

class SchemaDiscovery:
    def __init__(self, db_path: str):
        """
        Initialize with path to SQLite database.
        Example: 'sqlite:///path/to/db.sqlite'
        """
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.inspector = inspect(self.engine)

    def get_schema(self):
        """
        Returns a dictionary representing the database schema.
        {
            "table_name": [
                {"name": "col1", "type": "INTEGER", "primary_key": true},
                ...
            ],
            ...
        }
        """
        schema = {}
        for table_name in self.inspector.get_table_names():
            columns_info = []
            columns = self.inspector.get_columns(table_name)
            for col in columns:
                columns_info.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "primary_key": col.get("primary_key", False) == 1,
                    "nullable": col.get("nullable", True)
                })
            schema[table_name] = columns_info
        return schema

    def get_foreign_keys(self):
        """
        Returns a dictionary of foreign keys for each table.
        """
        fks = {}
        for table_name in self.inspector.get_table_names():
            fks[table_name] = self.inspector.get_foreign_keys(table_name)
        return fks

if __name__ == "__main__":
    # Test with a dummy file if needed, or intended to be imported.
    print("SchemaDiscovery module loaded.")
