# Text-to-SQL Application - Sequence Diagrams

This document contains detailed sequence diagrams showing the flow of the Text-to-SQL application.

## 1. Application Initialization Flow

This diagram shows what happens when the application starts up.

```mermaid
sequenceDiagram
    actor User
    participant Main as run_inference.py
    participant RAG as rag.py
    participant STModel as SentenceTransformer
    participant Loader as model_loader.py
    participant HF as HuggingFace Hub
    participant Model as Qwen2.5-Coder

    User->>Main: Run application
    activate Main
    
    Main->>Main: Load DUMMY_SCHEMAS
    
    Main->>RAG: initialize_retriever(schemas)
    activate RAG
    RAG->>STModel: SentenceTransformer('all-MiniLM-L6-v2')
    activate STModel
    STModel-->>RAG: retriever_model
    deactivate STModel
    
    RAG->>RAG: Store schemas in global variable
    RAG->>STModel: encode(schemas)
    activate STModel
    STModel-->>RAG: schema_embeddings (tensor)
    deactivate STModel
    RAG-->>Main: Initialization complete
    deactivate RAG
    
    Main->>Loader: load_model()
    activate Loader
    Loader->>HF: AutoTokenizer.from_pretrained()
    activate HF
    HF-->>Loader: tokenizer
    deactivate HF
    
    Loader->>HF: AutoModelForCausalLM.from_pretrained()
    activate HF
    HF->>HF: Download model if not cached (988MB)
    HF-->>Loader: model
    deactivate HF
    
    Loader->>Loader: Move model to GPU/CPU
    Loader-->>Main: (model, tokenizer)
    deactivate Loader
    
    Main-->>User: "Model loaded. Ready for queries."
    deactivate Main
```

## 2. Query Processing Flow (Main Inference)

This diagram shows the complete flow when a user submits a natural language query.

```mermaid
sequenceDiagram
    actor User
    participant Main as run_inference.py
    participant RAG as rag.py (retrieve_schema)
    participant STModel as SentenceTransformer
    participant Inference as inference.py
    participant Tokenizer as PreTrainedTokenizer
    participant Model as Qwen2.5-Coder Model
    
    User->>Main: Enter query: "Show me all users older than 25"
    activate Main
    
    Note over Main,RAG: Step 1: Schema Retrieval (RAG)
    Main->>RAG: retrieve_schema(query, top_k=1)
    activate RAG
    
    RAG->>STModel: encode(query) -> query_embedding
    activate STModel
    STModel-->>RAG: query_embedding (tensor)
    deactivate STModel
    
    RAG->>RAG: semantic_search(query_embedding, schema_embeddings)
    RAG->>RAG: Find top-k most similar schemas
    RAG-->>Main: "CREATE TABLE users (id INT, name TEXT, age INT...)"
    deactivate RAG
    
    Main->>User: Display retrieved context
    
    Note over Main,Model: Step 2: SQL Generation
    Main->>Inference: generate_sql(model, tokenizer, query, schema)
    activate Inference
    
    Inference->>Inference: Build prompt with Instruction/Context/Response format
    
    Inference->>Tokenizer: tokenizer(prompt, return_tensors="pt")
    activate Tokenizer
    Tokenizer->>Tokenizer: Convert text to token IDs
    Tokenizer-->>Inference: inputs (tensor on device)
    deactivate Tokenizer
    
    Inference->>Model: model.generate(inputs, max_new_tokens=512)
    activate Model
    Model->>Model: Run forward pass through transformer layers
    Model->>Model: Generate tokens autoregressively
    Model->>Model: Stop at max_tokens or EOS token
    Model-->>Inference: outputs (tensor of token IDs)
    deactivate Model
    
    Inference->>Tokenizer: tokenizer.decode(outputs, skip_special_tokens=True)
    activate Tokenizer
    Tokenizer->>Tokenizer: Convert token IDs back to text
    Tokenizer-->>Inference: Full response text
    deactivate Tokenizer
    
    Inference->>Inference: Post-process: Extract SQL after Response
    Inference-->>Main: SELECT * FROM users WHERE age greater than 25
    deactivate Inference
    
    Main->>User: Display generated SQL
    deactivate Main
```

## 3. RAG Retrieval Detail Flow

This diagram zooms into the semantic search mechanism used in RAG.

```mermaid
sequenceDiagram
    participant Client as Caller
    participant RAG as rag.py
    participant Model as SentenceTransformer
    participant Util as sentence_transformers.util
    
    Client->>RAG: retrieve_schema("query about users", top_k=1)
    activate RAG
    
    alt Retriever not initialized
        RAG-->>Client: Return empty string ""
    else Retriever initialized
        RAG->>Model: encode(query) -> Tensor
        activate Model
        Note over Model: Convert query to 384-dim vector
        Model-->>RAG: query_embedding [1, 384]
        deactivate Model
        
        RAG->>Util: semantic_search(query_emb, schema_embs, top_k=1)
        activate Util
        Note over Util: Compute cosine similarity<br/>between query and all schemas
        Util->>Util: Sort by similarity scores
        Util-->>RAG: [{corpus_id: 0, score: 0.87}, ...]
        deactivate Util
        
        loop For each hit in top_k results
            RAG->>RAG: Append schema_corpus[corpus_id] to results
        end
        
        RAG->>RAG: Join results with "\n\n"
        RAG-->>Client: "CREATE TABLE users (...)"
    end
    deactivate RAG
```

## 4. Model Inference Detail Flow

This diagram shows what happens inside the transformer model during generation.

```mermaid
sequenceDiagram
    participant Inference as inference.py
    participant Model as Qwen2.5 Model
    participant Attention as Multi-Head Attention
    participant FFN as Feed Forward Network
    participant Decoder as Output Decoder
    
    Inference->>Model: generate(inputs, max_new_tokens=512)
    activate Model
    
    Model->>Model: Embed input tokens -> [batch, seq_len, hidden_dim]
    
    loop For each new token (up to 512)
        Model->>Attention: Process through 24 transformer layers
        activate Attention
        
        loop 24 Layers
            Attention->>Attention: Self-Attention: Q, K, V computation
            Attention->>Attention: Attention weights = softmax(QK^T/√d)
            Attention->>Attention: Context = Attention × V
            Attention->>FFN: Pass to Feed Forward
            activate FFN
            FFN->>FFN: Linear -> ReLU -> Linear
            FFN-->>Attention: Transformed output
            deactivate FFN
            Attention->>Attention: Add & Norm (Residual Connection)
        end
        
        Attention-->>Model: Final layer output
        deactivate Attention
        
        Model->>Decoder: Project to vocabulary space
        activate Decoder
        Decoder->>Decoder: Linear(hidden_dim, vocab_size)
        Decoder->>Decoder: Get next token (greedy: argmax or sampling)
        Decoder-->>Model: next_token_id
        deactivate Decoder
        
        alt next_token is EOS
            Model->>Model: Stop generation
        else Continue generation
            Model->>Model: Append token to sequence
        end
    end
    
    Model-->>Inference: Complete sequence of token IDs
    deactivate Model
```

## 5. Evaluation Flow (Conceptual)

This diagram shows how evaluation would work (currently placeholder).

```mermaid
sequenceDiagram
    participant Script as run_eval.py
    participant Dataset as Spider/BirdBench
    participant Inference as inference.py
    participant Evaluator as evaluate.py
    participant Database as SQLite DB
    
    Script->>Dataset: Load test examples
    activate Dataset
    Dataset-->>Script: [{question, sql, db}, ...]
    deactivate Dataset
    
    loop For each test example
        Script->>Inference: generate_sql(model, tokenizer, question, schema)
        activate Inference
        Inference-->>Script: predicted_sql
        deactivate Inference
        
        par Exact Match Evaluation
            Script->>Evaluator: compute_exact_match(predicted, reference)
            activate Evaluator
            Evaluator->>Evaluator: Normalize and compare strings
            Evaluator-->>Script: exact_match_score (bool)
            deactivate Evaluator
        and Execution Accuracy Evaluation
            Script->>Evaluator: compute_execution_accuracy(predicted, reference, db)
            activate Evaluator
            Evaluator->>Database: Execute predicted SQL
            activate Database
            Database-->>Evaluator: result_1
            deactivate Database
            
            Evaluator->>Database: Execute reference SQL
            activate Database
            Database-->>Evaluator: result_2
            deactivate Database
            
            Evaluator->>Evaluator: Compare results
            Evaluator-->>Script: execution_accuracy (bool)
            deactivate Evaluator
        end
        
        Script->>Script: Aggregate scores
    end
    
    Script->>Script: Calculate final metrics
    Script->>User: Report: Exact Match: 65%, Execution: 72%
```

## 6. Error Handling Flow

This diagram shows error scenarios and how they're handled.

```mermaid
sequenceDiagram
    actor User
    participant Main as run_inference.py
    participant Loader as model_loader.py
    participant HF as HuggingFace
    
    User->>Main: Start application
    Main->>Loader: load_model()
    activate Loader
    
    Loader->>HF: AutoTokenizer.from_pretrained()
    activate HF
    
    alt Model Not Found
        HF-->>Loader: HTTPError / ModelNotFoundError
        Loader->>Loader: Catch Exception
        Loader->>Loader: Print error message
        Loader-->>Main: Raise Exception
        Main-->>User: Error: Model loading failed
    else Network Error
        HF-->>Loader: ConnectionError
        Loader->>Loader: Catch Exception
        Loader-->>Main: Raise Exception
        Main-->>User: Error: Check internet connection
    else Success
        HF-->>Loader: tokenizer
        deactivate HF
        Loader-->>Main: (model, tokenizer)
        Main-->>User: Success
    end
    deactivate Loader
```

## Component Interaction Summary

### Key Components:
1. **run_inference.py**: Entry point, orchestrates the flow
2. **rag.py**: Retrieves relevant database schemas using semantic search
3. **model_loader.py**: Downloads and initializes the Qwen model
4. **inference.py**: Core SQL generation logic
5. **evaluate.py**: Metrics computation (exact match, execution accuracy)

### Data Flow:
```
User Query 
  → RAG Retrieval (find relevant schema) 
  → Prompt Construction (Instruction + Context) 
  → Tokenization (text → numbers) 
  → Model Inference (transformer processing) 
  → Detokenization (numbers → text) 
  → Post-processing (extract SQL) 
  → Generated SQL
```
