# Project Overview: Cognitive Bridge

## Executive Summary
The **Cognitive Bridge** project is a "Double-Translation" system designed to bridge the gap between natural language and diverse database paradigms. It enables users to query databases using natural language, translating requests first into SQL (for relational structure) and then into MQL (MongoDB Query Language) for NoSQL document stores.

## Key Features
- **Natural Language to SQL**: Converts user questions into executable SQL queries using the **Qwen2.5-Coder-3B** model.
- **SQL to NoSQL Transformation**: Transpiles generated SQL queries into efficient MongoDB aggregation pipelines.
- **RAG-Enhanced Interpretation**: Uses Retrieval-Augmented Generation to link natural language terms to specific database schema elements.
- **Reflexion Loop**: Implements a self-correction mechanism where the model analyzes SQL execution errors and iteratively refines the query.
- **Auto-Migration**: Utilities to migrate data and schema from relational (SQLite) sources to document-based (MongoDB) storage.

## Technology Stack
| Component | Technology | Port | Description |
| :--- | :--- | :--- | :--- |
| **Language** | Python 3.10+ | - | Core application logic. |
| **Frontend** | Streamlit | 8501 | Interactive web UI for query input and result visualization. |
| **Backend** | FastAPI | 8080 | REST API orchestrator managing the transaction flow. |
| **Inference** | vLLM | 8000 | High-performance LLM serving with OpenAI-compatible API. |
| **Databases** | SQLite, MongoDB | 27018* | Source relational data (SQLite) and target NoSQL store (MongoDB). |
| **Containerization** | Docker Compose | - | Full-stack deployment with 4 containerized services. |

*MongoDB external port 27018 maps to internal container port 27017.

## Gen AI Stack

### Small Language Model (SLM)
- **Base Model**: `Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4`
- **Quantization**: 4-bit GPTQ quantization for efficient inference on consumer GPUs (8GB VRAM).
- **LoRA Adapter**: `Qwen2.5-Coder-3B-Instruct-mql-adapter` for SQL-to-MQL transpilation.
- **Justification**: A highly capable yet lightweight coding model that balances SQL generation accuracy with deployment costs. The finetuned LoRA adapter specializes in MongoDB query translation.

### Retrieval-Augmented Generation (RAG)
- **Embedding Model**: `all-MiniLM-L6-v2` (Sentence Transformers).
- **Strategy**: Schema-based retrieval.
    - **Indexing**: Database tables are indexed as text documents (`"Table: {name}. Columns: {col1}, {col2}..."`).
    - **Retrieval**: Cosine similarity is used to link natural language terms in the question to the most relevant database tables.
    - **Top-K**: The system typically retrieves the top 3-5 tables to reduce context window usage.

### Fine-Tuning Details
- **Method**: QLoRA (Quantized Low-Rank Adaptation).
- **Dataset**: `sql_to_mql_finetuning.jsonl` (Custom instruction set for SQL-to-MQL conversion).
- **LoRA Configuration**:
    - **Rank (r)**: 64
    - **Alpha**: 16
    - **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
    - **Max LoRA Rank**: 64 (configured in vLLM)
- **Training Environment**: Google Colab (T4 GPU) or GCP Vertex AI.
- **Hyperparameters**:
    - **Epochs**: 1
    - **Learning Rate**: 2e-4
    - **Optimizer**: `paged_adamw_8bit`
- **Deployment**: Adapter mounted at `./Qwen2.5-Coder-3B-Instruct-mql-adapter` and served via vLLM with `--enable-lora` flag.

## Directory Structure
- **`backend/`**: FastAPI application and API infrastructure.
  - **`app/main.py`**: FastAPI entry point with `/api/v1/generate` endpoint.
  - **`app/schemas.py`**: Pydantic models for request/response validation.
  - **`app/services/orchestrator.py`**: Core orchestration logic (283 lines).
- **`frontend/`**: Streamlit application.
  - **`app.py`**: Main UI with chat interface, result visualization, and database selector.
- **`services/`**: Shared business logic libraries.
  - **`inference/rag_service.py`**: RAG implementation using `all-MiniLM-L6-v2` for schema linking.
  - **`migration/migrate.py`**: MigrationService class for SQLite-to-MongoDB data transfer.
  - **`migration/schema_discovery.py`**: SchemaDiscovery class for metadata extraction.
- **`data/`**: Storage for BirdBench datasets and SQLite database files.
- **`scripts/`**: 20+ utility scripts for data preparation, migration, and finetuning.
- **`Qwen2.5-Coder-3B-Instruct-mql-adapter/`**: Finetuned LoRA adapter for SQL-to-MQL conversion.
- **`docker-compose.yml`**: Multi-container orchestration (4 services).
- **`documentation/`**: Architecture, design, and sequence diagrams.
