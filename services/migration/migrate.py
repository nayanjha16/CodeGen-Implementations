import os
import pandas as pd
from pymongo import MongoClient, ASCENDING
from sqlalchemy import create_engine
from .schema_discovery import SchemaDiscovery
import datetime
import dateutil.parser

class MigrationService:
    def __init__(self, sqlite_path: str, mongo_uri: str, db_name: str):
        self.sqlite_path = sqlite_path
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.discovery = SchemaDiscovery(sqlite_path)
        self.engine = create_engine(f"sqlite:///{sqlite_path}")

    def infer_and_convert_types(self, df: pd.DataFrame, columns_info: list):
        """
        Convert DataFrame columns to appropriate types for MongoDB.
        """
        for col_info in columns_info:
            col_name = col_info['name']
            if col_name not in df.columns:
                continue

            # Attempt date conversion for text fields that look like dates
            if col_info['type'] in ['TEXT', 'VARCHAR', 'CHAR']:
                # Sample a few non-null values to check if they look like dates
                sample = df[col_name].dropna().head(10)
                if not sample.empty:
                    try:
                        # rigorous check: see if all samples are dates
                        is_date = True
                        for val in sample:
                            if not isinstance(val, str) or len(val) < 10: # Simple heuristic
                                is_date = False
                                break
                            try:
                                dateutil.parser.parse(val)
                            except:
                                is_date = False
                                break
                        
                        # DISABLED: Datetime conversion causes NaT serialization errors
                        # if is_date:
                        #     print(f"Converting column {col_name} to datetime objects.")
                        #     df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                        #     # Convert NaT to None for MongoDB compatibility
                        #     df[col_name] = df[col_name].apply(lambda x: None if pd.isna(x) else x)
                    except Exception:
                        pass
        
        # Convert NaN to None for Mongo (null)
        df = df.where(pd.notnull(df), None)
        return df

    def migrate(self):
        print(f"Starting migration for {self.sqlite_path} -> {self.db.name}")
        schema = self.discovery.get_schema()
        fks = self.discovery.get_foreign_keys()

        for table_name, columns in schema.items():
            print(f"Migrating table: {table_name}")
            
            # Read Data in Chunks to handle large tables (e.g. financial)
            chunk_size = 50000
            total_records = 0
            
            # Clear existing data first
            self.db[table_name].delete_many({})

            try:
                # Use chunksize to get an iterator
                # CRITICAL: parse_dates=False prevents NaT errors by keeping dates as strings
                for chunk_df in pd.read_sql_table(table_name, self.engine, chunksize=chunk_size, parse_dates=False):
                    if chunk_df.empty:
                        continue
                        
                    # Type Inference on the chunk
                    chunk_df = self.infer_and_convert_types(chunk_df, columns)
                    
                    # CRITICAL FIX: Convert datetime columns to strings to avoid NaT errors
                    for col in chunk_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(chunk_df[col]):
                            print(f"Converting datetime column {col} to string...")
                            chunk_df[col] = chunk_df[col].astype(str).replace('NaT', None)
                    
                    # Insert to Mongo
                    records = chunk_df.to_dict(orient='records')
                    if records:
                        self.db[table_name].insert_many(records)
                        total_records += len(records)
                        print(f"Inserted chunk of {len(records)} records into {table_name} (Total: {total_records})")
                
                if total_records == 0:
                     print(f"Table {table_name} is empty.")
                else:
                     print(f"Finished migrating {table_name}. Total records: {total_records}")

            except Exception as e:
                print(f"Error migrating table {table_name}: {e}")

            # Indexing based on Foreign Keys
            if table_name in fks:
                for fk in fks[table_name]:
                    constrained_columns = fk['constrained_columns']
                    # SQLite foreign keys usually single column but can be composite
                    # Mongo Create Index
                    index_fields = [(col, ASCENDING) for col in constrained_columns]
                    if index_fields:
                        print(f"Creating index on {table_name} for FK: {constrained_columns}")
                        self.db[table_name].create_index(index_fields)

        print("Migration complete.")

if __name__ == "__main__":
    # Example Usage
    # service = MigrationService("path/to/db.sqlite", "mongodb://localhost:27017", "my_migrated_db")
    # service.migrate()
    pass
