# Sequence Diagrams

## 1. End-to-End Query Execution
This diagram depicts the flow when a user asks a question via the UI.

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI<br/>:8501
    participant API as FastAPI Backend<br/>:8080
    participant RAG as RAG Service
    participant Sbert as SentenceTransformer
    participant LLM as vLLM Engine<br/>:8000
    participant SQL as SQLite DB
    participant Mongo as MongoDB<br/>:27018

    User->>UI: Enters "How many schools are there?"
    UI->>API: POST /api/v1/generate<br/>{question, db_id: "california_schools"}
    
    activate API
    
    note over API: 1. Schema Discovery
    API->>SQL: Read metadata via SchemaDiscovery
    SQL-->>API: Table schemas, columns, FKs
    
    note over API: 2. Schema Linking (RAG)
    API->>RAG: ingest_schema(db_id, schema)
    RAG->>Sbert: encode(table descriptions)
    Sbert-->>RAG: embeddings cached
    
    API->>RAG: retrieve_relevant_tables(question)
    RAG->>Sbert: encode(question)
    Sbert-->>RAG: question_embedding
    RAG->>RAG: compute cosine similarity
    RAG-->>API: ["schools" (top-k tables)]
    
    note over API,LLM: 3. NL → SQL Generation
    API->>LLM: POST /v1/chat/completions<br/>model: "Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4"<br/>prompt: Schema + Question
    LLM-->>API: "SELECT COUNT(*) FROM schools"
    
    note over API: 4. SQL Execution & Validation
    API->>SQL: Execute SQL via SQLAlchemy
    alt Success
        SQL-->>API: Result: [{"COUNT(*)": 17686}]
    else Error
        SQL-->>API: SQL Syntax Error
        note over API: (Future: Reflexion loop retry)
    end
    
    note over API,LLM: 5. SQL → NoSQL Transpilation
    API->>LLM: POST /v1/chat/completions<br/>model: "mql-adapter" (LoRA)<br/>prompt: "Convert SQL to MongoDB"
    LLM-->>API: [{"$count": "total_schools"}]
    
    note over API: 6. Extract Start Collection
    API->>API: _extract_start_table(sql)<br/>→ "schools"
    
    note over API: 7. NoSQL Execution
    API->>Mongo: db["california_schools"]["schools"]<br/>.aggregate(pipeline)
    Mongo-->>API: Result: [{"total_schools": 17686}]
    
    note over API: 8. Result Comparison
    API->>API: _compare_results(sql_res, mongo_res)<br/>Normalize types, set comparison
    API->>API: execution_match = True
    
    API-->>UI: QueryResponse JSON<br/>{sql_query, mongo_pipeline,<br/>sql_result, mongo_result,<br/>execution_match: true}
    deactivate API
    
    UI->>User: Display results:<br/>✅ Execution Match<br/>SQL: COUNT(*) = 17686<br/>MQL: total_schools = 17686
```

---

## 2. Model Selection Flow (LoRA Adapter)

This diagram shows how the orchestrator selects between the base model and LoRA adapter.

```mermaid
sequenceDiagram
    participant Orch as Orchestrator
    participant LLM as vLLM Engine
    participant Base as Base Model<br/>Qwen2.5-Coder-3B
    participant LoRA as LoRA Adapter<br/>mql-adapter

    note over Orch: Step 1: SQL Generation
    Orch->>LLM: POST /v1/chat/completions<br/>model="Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4"
    LLM->>Base: Route to base model
    Base-->>LLM: Generated SQL
    LLM-->>Orch: SQL query string
    
    note over Orch: Step 2: MQL Transpilation
    Orch->>LLM: POST /v1/chat/completions<br/>model="mql-adapter"
    LLM->>LoRA: Route to LoRA adapter
    LoRA->>Base: Apply LoRA weights to base
    LoRA-->>LLM: Generated MQL pipeline
    LLM-->>Orch: MongoDB aggregation JSON
    
    note over Orch: LoRA adapter specializes<br/>in SQL→MQL translation
```

---

## 3. Data Migration Process
This diagram shows how data is moved from the source SQLite to the target MongoDB.

```mermaid
sequenceDiagram
    participant Script as Migration Script
    participant Svc as MigrationService
    participant Disc as SchemaDiscovery
    participant SQL as SQLite Source
    participant Mongo as MongoDB Target

    Script->>Svc: MigrationService(db_path, mongo_uri, db_name)
    Script->>Svc: migrate()
    activate Svc
    
    Svc->>Disc: get_schema()
    Disc->>SQL: inspect.get_table_names()
    Disc->>SQL: inspect.get_columns(table)
    SQL-->>Disc: Column metadata
    Disc-->>Svc: {table: [columns_info]}
    
    loop For each Table
        Svc->>Mongo: db[table].delete_many({})
        note over Mongo: Clear existing data
        
        loop Read in 50k-record chunks
            Svc->>SQL: pd.read_sql_table(table, chunksize=50000)
            SQL-->>Svc: DataFrame chunk
            
            Svc->>Svc: infer_and_convert_types(df, columns)<br/>Handle NaN→None, datetime→string
            
            Svc->>Mongo: db[table].insert_many(records)
            Mongo-->>Svc: Acknowledged
            note over Svc: Log: "Inserted 50000 records"
        end
    end
    
    Svc->>Disc: get_foreign_keys()
    Disc-->>Svc: {table: [fk_constraints]}
    
    loop For each Foreign Key
        Svc->>Mongo: db[table].create_index([(fk_col, ASCENDING)])
        note over Mongo: Index for query performance
    end
    
    Svc-->>Script: Migration Complete
    deactivate Svc
```

---

## 4. Docker Compose Startup Sequence

This diagram shows the service initialization order and dependencies.

```mermaid
sequenceDiagram
    participant User
    participant Docker as Docker Compose
    participant Mongo as mongo-db
    participant Inference as inference-engine
    participant Backend as backend-api
    participant Frontend as frontend-ui
    
    User->>Docker: docker-compose up -d
    
    note over Docker: Start independent services
    par Start mongo-db
        Docker->>Mongo: docker run mongo:latest<br/>Port 27018:27017
        Mongo->>Mongo: Initialize database
        Mongo-->>Docker: Ready
    and Start inference-engine
        Docker->>Inference: docker run vllm/vllm-openai:latest<br/>Port 8000:8000
        Inference->>Inference: Load Qwen2.5-Coder-3B-GPTQ-Int4
        Inference->>Inference: Mount LoRA adapter (mql-adapter)
        note over Inference: GPU allocation: 85%<br/>Max model length: 8192
        Inference-->>Docker: Ready (serving on :8000)
    end
    
    note over Docker: Wait for dependencies
    Docker->>Backend: docker run backend-api<br/>depends_on: [mongo-db, inference-engine]
    Backend->>Mongo: Test connection (MONGO_URI)
    Mongo-->>Backend: Connected
    Backend->>Inference: Test /v1/models endpoint
    Inference-->>Backend: {"data": ["base", "mql-adapter"]}
    Backend->>Backend: Initialize Orchestrator & RAG Service
    Backend-->>Docker: Ready (serving on :8080)
    
    Docker->>Frontend: docker run frontend-ui<br/>depends_on: [backend-api]
    Frontend->>Backend: GET / (health check)
    Backend-->>Frontend: {"status": "ok"}
    Frontend-->>Docker: Ready (serving on :8501)
    
    Docker-->>User: All services running<br/>Access: http://localhost:8501
    
    note over User: User can now access the UI<br/>and submit queries
```

---

## 5. Error Handling and Reflexion (Future Enhancement)

```mermaid
sequenceDiagram
    participant Orch as Orchestrator
    participant LLM as vLLM Engine
    participant SQL as SQLite DB
    
    Orch->>LLM: Generate SQL (Attempt 1)
    LLM-->>Orch: "SELECT * FORM users"
    
    Orch->>SQL: Execute SQL
    SQL-->>Orch: ❌ Error: syntax error near "FORM"
    
    note over Orch: Prepare error feedback
    Orch->>LLM: Regenerate SQL<br/>Error: "syntax error near FORM"<br/>Previous: "SELECT * FORM users"
    LLM-->>Orch: "SELECT * FROM users"
    
    Orch->>SQL: Execute SQL (Attempt 2)
    SQL-->>Orch: ✅ Success: [records]
    
    note over Orch: Proceed to MQL generation
```

**Note**: Current implementation has basic error catching but no automatic retry loop. Future versions will implement full reflexion as shown above.
