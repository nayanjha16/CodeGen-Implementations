Important Notes:

RAG is currently only implemented for the SQLite schema, not separately for MongoDB. Here's why this design works:

Current Flow:

User Question → RAG retrieves relevant SQLite tables (using vector similarity)
Filtered SQLite Schema → LLM generates SQL query
SQL Query → LLM converts to MongoDB Pipeline (MQL)
Why MongoDB doesn't need separate RAG:

MongoDB collections have the exact same structure as SQLite tables (we migrated them 1:1)
The SQLite schema information is reused for MQL generation (see line 102 in 
orchestrator.py
: mongo_schema=json.dumps(schema, indent=2))
Since MQL is derived from SQL, not from scratch, the schema context is already filtered
Potential Enhancement (not implemented): If you wanted to support MongoDB-native queries (bypassing SQL entirely), you'd need:

A separate RAG index for MongoDB collection schemas
Direct NL → MQL translation (skipping the SQL step)
But for the current SQL-to-NoSQL translation pipeline, the single SQLite-based RAG is sufficient because both databases share the same logical schema.


API KEYs:

https://aistudio.google.com/app/api-keys

Capstone Project - API Key: AIzaSyChH7ygjUBycMrv-WxxuLOE3znAtdtwuHE
The above API Key is linked to: 'My Billing Account'.
More info @ console.cloud.google.com.
Account name
My Billing Account, 01D480-5B4FE8-3B9503

SQL Query to NOSQL query for finetuning purposes was created using gemini-2.0-flash API calls

