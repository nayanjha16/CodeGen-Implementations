# Cognitive Bridge: Text-to-SQL-to-NoSQL

**Architecting a High-Fidelity Transformation Pipeline Using Small Language Models**

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)](https://streamlit.io/)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Deployment](#detailed-deployment)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)

---

## Overview

**Cognitive Bridge** implements a "Double-Translation" pipeline that bridges natural language and diverse database paradigms (Relational & NoSQL). Using the **BirdBench** dataset and **Qwen2.5-Coder-3B** with a finetuned LoRA adapter, the system performs:

1. **NL → SQL**: Natural language to SQL query generation grounded in strict schema definitions
2. **SQL → MQL**: SQL to MongoDB Query Language transpilation using finetuned adapter

### Key Features
✨ **RAG-Enhanced Schema Linking**: Retrieves top-K relevant tables using embeddings  
🔄 **Dual Database Execution**: Runs queries on both SQLite and MongoDB  
✅ **Result Verification**: Validates query equivalence across database paradigms  
🚀 **LoRA Finetuned Model**: Specialized SQL-to-MQL transpilation adapter  
🐳 **Dockerized Deployment**: Complete 4-service containerized stack

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐      ┌──────────────────┐              │
│  │  Streamlit UI   │─────▶│   FastAPI API    │              │
│  │    :8501        │      │     :8080        │              │
│  └─────────────────┘      └────────┬─────────┘              │
│                                     │                        │
│                          ┌──────────▼──────────┐             │
│                          │   Orchestrator      │             │
│                          └──────────┬──────────┘             │
│                                     │                        │
│              ┌──────────────────────┼──────────────────┐     │
│              │                      │                  │     │
│      ┌───────▼────────┐    ┌───────▼────────┐  ┌─────▼────┐│
│      │   RAG Service  │    │  vLLM Engine   │  │ MongoDB  ││
│      │  (Embeddings)  │    │  :8000         │  │  :27018  ││
│      └────────────────┘    │  Base + LoRA   │  └──────────┘│
│                            └────────────────┘               │
│                                                              │
│  SQLite DBs (BirdBench) ◀──────────────────────────────────┤
└──────────────────────────────────────────────────────────────┘
```

### Technology Stack
| Component | Technology | Port | Description |
|:----------|:-----------|:----:|:------------|
| **Frontend** | Streamlit | 8501 | Interactive chat UI |
| **Backend** | FastAPI | 8080 | REST API orchestrator |
| **Inference** | vLLM | 8000 | LLM serving (base + LoRA) |
| **Database** | MongoDB | 27018 | NoSQL document store |
| **Source Data** | SQLite | - | BirdBench datasets |

---

## Prerequisites

### Required Software
- **Docker Desktop** (v20.10+) with Docker Compose
- **NVIDIA GPU** (8GB VRAM recommended) with drivers installed
- **NVIDIA Container Toolkit** for GPU support in Docker
- **Git** (for cloning the repository)

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 20GB free space
- **GPU**: NVIDIA GPU with CUDA support (for vLLM)

### Verify Prerequisites
```bash
# Check Docker
docker --version
docker compose version

# Check NVIDIA GPU (Linux/WSL)
nvidia-smi

# Check Git
git --version
```

---

## Quick Start

### 1️⃣ Clone Repository
```bash
git clone <repository-url>
cd text-2-sql-2-nosql
```

### 2️⃣ Set Environment Variables (Optional)
Create a `.env` file in the project root:
```bash
# Optional: Only needed for private Hugging Face models
HF_TOKEN=your_huggingface_token_here
```

### 3️⃣ Start All Services
```bash
docker compose up -d
```

This single command will:
- Pull all required Docker images
- Build custom containers (backend, frontend)
- Start all 4 services in the correct order
- Expose ports: 8501 (UI), 8080 (API), 8000 (vLLM), 27018 (MongoDB)

### 4️⃣ Access the Application
Open your browser and navigate to:
```
http://localhost:8501
```

**First-time startup**: The vLLM inference engine will download the Qwen2.5-Coder-3B model (~2GB). This takes 5-10 minutes on first run.

---

## Detailed Deployment

### Step-by-Step Setup

#### 1. Download BirdBench Data
```bash
# Install Python dependencies
pip install -r requirements.txt

# Download BirdBench datasets
python setup_data.py
```

This will download SQLite databases to `./data/minidev/MINIDEV/databases/`

#### 2. Prepare LoRA Adapter
The finetuned LoRA adapter should be in:
```
./Qwen2.5-Coder-3B-Instruct-mql-adapter/
```

If you need to finetune your own adapter, use:
```bash
# Run finetuning notebook
jupyter notebook finetune_qwen_colab.ipynb
```

#### 3. Start Services Individually (For Development)

**Option A: Start All Services**
```bash
docker compose up -d
```

**Option B: Start Services Selectively**
```bash
# Start only MongoDB
docker compose up -d mongo-db

# Start inference engine (requires GPU)
docker compose up -d inference-engine

# Start backend (requires mongo + inference)
docker compose up -d backend-api

# Start frontend (requires backend)
docker compose up -d frontend-ui
```

#### 4. Migrate Data to MongoDB
```bash
# Migrate a specific database
python scripts/migrate_all.py

# Or use the simple migration script
python scripts/simple_migrate.py
```

#### 5. Verify Deployment
```bash
# Check all services are running
docker compose ps

# Expected output:
# NAME               STATUS    PORTS
# backend-api        Up        0.0.0.0:8080->8080/tcp
# frontend-ui        Up        0.0.0.0:8501->8501/tcp
# inference-engine   Up        0.0.0.0:8000->8000/tcp
# mongo-db           Up        0.0.0.0:27018->27017/tcp
```

#### 6. Check Service Health
```bash
# Test backend API
curl http://localhost:8080/

# Test vLLM inference
curl http://localhost:8000/v1/models

# Test MongoDB connection
docker exec -it mongo-db mongosh --eval "db.runCommand({ ping: 1 })"
```

---

## Configuration

### Docker Compose Services

#### Frontend (Streamlit)
```yaml
frontend-ui:
  build: ./frontend
  ports:
    - "8501:8501"
  environment:
    - BACKEND_URL=http://backend-api:8080
  volumes:
    - ./frontend:/app
```

#### Backend (FastAPI)
```yaml
backend-api:
  build: ./backend
  ports:
    - "8080:8080"
  environment:
    - MONGO_URI=mongodb://mongo-db:27017
    - INFERENCE_URL=http://inference-engine:8000/v1
  volumes:
    - ./backend:/app
    - ./data:/app/data
    - ./services:/app/services
```

#### Inference Engine (vLLM)
```yaml
inference-engine:
  image: vllm/vllm-openai:latest
  ports:
    - "8000:8000"
  shm_size: '4gb'
  command: >
    --model Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4
    --quantization gptq
    --dtype float16
    --gpu-memory-utilization 0.85
    --max-model-len 8192
    --enable-lora
    --lora-modules mql-adapter=/root/adapters/mql-adapter
    --max-lora-rank 64
  volumes:
    - ~/.cache/huggingface:/root/.cache/huggingface
    - ./Qwen2.5-Coder-3B-Instruct-mql-adapter:/root/adapters/mql-adapter
```

### Customization

**Adjust GPU Memory** (if you have limited VRAM):
Edit `docker-compose.yml`:
```yaml
--gpu-memory-utilization 0.7  # Reduce from 0.85 to 0.7
--max-model-len 4096          # Reduce from 8192 to 4096
```

**Change MongoDB Port**:
```yaml
mongo-db:
  ports:
    - "27017:27017"  # Use standard MongoDB port
```

**Enable Debug Logging**:
Add to backend environment:
```yaml
environment:
  - LOG_LEVEL=DEBUG
```

---

## Usage

### Basic Query Flow

1. **Access UI**: Navigate to `http://localhost:8501`
2. **Select Database**: Choose from dropdown (e.g., `california_schools`)
3. **Ask Question**: Type natural language query: "How many schools are there?"
4. **View Results**: See SQL and MongoDB results side-by-side

### Example Queries

**california_schools database:**
```
- How many schools are there?
- Show all schools in Alameda
- What is the average enrollment by city?
- List schools with more than 1000 students
```

**financial database:**
```
- What is the total transaction amount?
- Show all accounts with negative balance
- List top 5 customers by transaction count
```

### API Usage

**Direct API Call:**
```bash
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many schools are there?",
    "db_id": "california_schools"
  }'
```

**Response:**
```json
{
  "sql_query": "SELECT COUNT(*) FROM schools",
  "mongo_pipeline": [{"$count": "total"}],
  "sql_result": [{"COUNT(*)": 17686}],
  "mongo_result": [{"total": 17686}],
  "execution_match": true,
  "error": null
}
```

---

## Project Structure

```
text-2-sql-2-nosql/
├── backend/                          # FastAPI application
│   ├── app/
│   │   ├── main.py                   # API entry point (19 lines)
│   │   ├── schemas.py                # Pydantic models (24 lines)
│   │   └── services/
│   │       └── orchestrator.py       # Core logic (283 lines)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                         # Streamlit application
│   ├── app.py                        # UI code (131 lines)
│   ├── Dockerfile
│   └── requirements.txt
├── services/                         # Shared services
│   ├── inference/
│   │   └── rag_service.py           # RAG implementation (70 lines)
│   └── migration/
│       ├── schema_discovery.py      # Schema extraction (50 lines)
│       └── migrate.py               # Data migration (121 lines)
├── scripts/                          # Utility scripts (20+ files)
│   ├── migrate_all.py
│   ├── simple_migrate.py
│   └── ...
├── data/                             # BirdBench datasets
│   └── minidev/MINIDEV/databases/   # SQLite files
├── Qwen2.5-Coder-3B-Instruct-mql-adapter/  # LoRA adapter
├── documentation/                    # Architecture docs
│   ├── project_overview.md
│   ├── high_level_architecture.md
│   ├── high_level_design.md
│   ├── low_level_design.md
│   └── sequence_diagrams.md
├── docker-compose.yml                # Multi-container orchestration
├── requirements.txt                  # Root dependencies
└── README.md                         # This file
```

---

## Documentation

### Architecture & Design
- **[Project Overview](./documentation/project_overview.md)**: Executive summary, tech stack, Gen AI details
- **[High-Level Architecture](./documentation/high_level_architecture.md)**: System diagram with components
- **[High-Level Design](./documentation/high_level_design.md)**: C4 container diagram, Docker deployment
- **[Low-Level Design](./documentation/low_level_design.md)**: Class diagrams, algorithms
- **[Sequence Diagrams](./documentation/sequence_diagrams.md)**: Query flow, migration process

### Technical Specifications
- **Model**: Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4 (4-bit quantized)
- **LoRA Adapter**: Custom finetuned `mql-adapter` for SQL-to-MQL transpilation
- **RAG Embedding**: all-MiniLM-L6-v2 for schema linking
- **Datasets**: BirdBench minidev (11 databases)

---

## Troubleshooting

### Common Issues

#### 1. vLLM Fails to Start (GPU Error)
```bash
# Check NVIDIA GPU is available
nvidia-smi

# Install NVIDIA Container Toolkit
# Ubuntu/Debian:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2. Out of Memory Error
Reduce GPU memory usage in `docker-compose.yml`:
```yaml
--gpu-memory-utilization 0.6
--max-model-len 4096
```

#### 3. MongoDB Connection Failed
```bash
# Check MongoDB is running
docker compose ps mongo-db

# Restart MongoDB
docker compose restart mongo-db

# Check logs
docker compose logs mongo-db
```

#### 4. Backend Can't Connect to Services
```bash
# Check all services are up
docker compose ps

# Verify network
docker network inspect text-2-sql-2-nosql_default

# Restart backend
docker compose restart backend-api
```

#### 5. Frontend Shows "Connection Error"
```bash
# Check backend is healthy
curl http://localhost:8080/

# Check BACKEND_URL environment variable
docker compose exec frontend-ui env | grep BACKEND_URL

# View frontend logs
docker compose logs frontend-ui
```

### Viewing Logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs inference-engine
docker compose logs backend-api
docker compose logs frontend-ui

# Follow logs in real-time
docker compose logs -f
```

### Stopping Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v

# Stop specific service
docker compose stop backend-api
```

---

## Performance Notes

### First-Time Startup
- **Model Download**: 5-10 minutes (Qwen2.5-Coder-3B ~2GB)
- **Model Loading**: 30-60 seconds
- **Total Startup**: ~10 minutes

### Subsequent Startups
- **Model Loading**: 30-60 seconds (from cache)
- **Service Ready**: < 2 minutes

### Query Performance
- **SQL Generation**: 2-5 seconds
- **MQL Transpilation**: 2-4 seconds
- **Total Query Time**: 5-15 seconds

---

## Contributing

This is a capstone project. For issues or questions, please contact the project maintainer.

---

## License

This project is part of an academic capstone. All rights reserved.

---

## Acknowledgments

- **BirdBench Dataset**: Cross-domain text-to-SQL benchmark
- **Qwen Team**: Qwen2.5-Coder-3B model
- **vLLM**: High-performance LLM serving
- **FastAPI & Streamlit**: Web framework and UI library

---

## Contact & Support

For technical questions about this project, please refer to the documentation in the `documentation/` folder.

**Happy Querying! 🚀**
