# Text-to-SQL Application - Sequence Diagrams

This document contains detailed sequence diagrams showing the flow of the Text-to-SQL application.

## 1. Application Initialization Flow

This diagram shows what happens when the application starts up.

```mermaid
sequenceDiagram
    actor User
    participant Main as run_inference.py (Deprecated)
    participant RAG as rag.py
    participant STModel as SentenceTransformer
    participant Loader as model_loader.py
    participant HF as HuggingFace Hub
    participant Model as Qwen2.5-Coder

    User->>Main: Run application
    activate Main
    
    Main->>Main: Load DUMMY_SCHEMAS
    
    Main->>RAG: initialize_retriever schemas
    activate RAG
    RAG->>STModel: SentenceTransformer all-MiniLM-L6-v2
    activate STModel
    STModel-->>RAG: retriever_model
    deactivate STModel
    
    RAG->>RAG: Store schemas in global variable
    RAG->>STModel: encode schemas
    activate STModel
    STModel-->>RAG: schema_embeddings tensor
    deactivate STModel
    RAG-->>Main: Initialization complete
    deactivate RAG
    
    Main->>Loader: load_model
    activate Loader
    Loader->>HF: AutoTokenizer.from_pretrained
    activate HF
    HF-->>Loader: tokenizer
    deactivate HF
    
    Loader->>HF: AutoModelForCausalLM.from_pretrained
    activate HF
    HF->>HF: Download model if not cached
    HF-->>Loader: model
    deactivate HF
    
    Loader->>Loader: Check CUDA availability
    Loader->>Loader: Move model to GPU or CPU
    Loader-->>Main: model and tokenizer
    deactivate Loader
    
    Main-->>User: Model loaded Ready for queries
    deactivate Main
```

## 2. Query Processing Flow - Main Inference

This diagram shows the complete flow when a user submits a natural language query.

```mermaid
sequenceDiagram
    actor User
    participant Main as run_inference.py
    participant RAG as rag.py
    participant STModel as SentenceTransformer
    participant Inference as inference.py
    participant Tokenizer as PreTrainedTokenizer
    participant Model as Qwen2.5-Coder
    
    User->>Main: Enter query Show me all users older than 25
    activate Main
    
    Note over Main,RAG: Step 1 Schema Retrieval RAG
    Main->>RAG: retrieve_schema query top_k=1
    activate RAG
    
    RAG->>STModel: encode query
    activate STModel
    STModel-->>RAG: query_embedding tensor
    deactivate STModel
    
    RAG->>RAG: semantic_search query vs schemas
    RAG->>RAG: Find top-k most similar schemas
    RAG-->>Main: CREATE TABLE users id name age email
    deactivate RAG
    
    Main->>User: Display retrieved context
    
    Note over Main,Model: Step 2 SQL Generation
    Main->>Inference: generate_sql model tokenizer query schema
    activate Inference
    
    Inference->>Inference: Build prompt with Instruction Schema Question format
    
    Inference->>Tokenizer: tokenizer prompt return_tensors pt
    activate Tokenizer
    Tokenizer->>Tokenizer: Convert text to token IDs
    Tokenizer-->>Inference: inputs tensor on device
    deactivate Tokenizer
    
    Inference->>Model: model.generate inputs max_new_tokens=512
    activate Model
    Model->>Model: Run forward pass through transformer layers
    Model->>Model: Generate tokens autoregressively
    Model->>Model: Stop at max_tokens or EOS token
    Model-->>Inference: outputs tensor of token IDs
    deactivate Model
    
    Inference->>Tokenizer: tokenizer.decode outputs
    activate Tokenizer
    Tokenizer->>Tokenizer: Convert token IDs back to text
    Tokenizer-->>Inference: Full response text
    deactivate Tokenizer
    
    Inference->>Inference: Post-process Extract SQL after Response
    Inference->>Inference: Remove markdown code blocks
    Inference->>Inference: Keep only first statement
    Inference-->>Main: SELECT FROM users WHERE age greater than 25
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
    participant Util as semantic_search util
    
    Client->>RAG: retrieve_schema query top_k=1
    activate RAG
    
    alt Retriever not initialized
        RAG-->>Client: Return empty string
    else Retriever initialized
        RAG->>Model: encode query to Tensor
        activate Model
        Note over Model: Convert query to 384-dim vector
        Model-->>RAG: query_embedding 1x384
        deactivate Model
        
        RAG->>Util: semantic_search query_emb schema_embs top_k=1
        activate Util
        Note over Util: Compute cosine similarity between query and all schemas
        Util->>Util: Sort by similarity scores
        Util-->>RAG: corpus_id 0 score 0.87
        deactivate Util
        
        loop For each hit in top_k results
            RAG->>RAG: Append schema to results
        end
        
        RAG->>RAG: Join results
        RAG-->>Client: CREATE TABLE users schema
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
    
    Inference->>Model: generate inputs max_new_tokens=512
    activate Model
    
    Model->>Model: Embed input tokens to hidden_dim
    
    loop For each new token up to 512
        Model->>Attention: Process through 24 transformer layers
        activate Attention
        
        loop 24 Layers
            Attention->>Attention: Self-Attention Q K V computation
            Attention->>Attention: Attention weights softmax QKT
            Attention->>Attention: Context equals Attention times V
            Attention->>FFN: Pass to Feed Forward
            activate FFN
            FFN->>FFN: Linear then SwiGLU then Linear
            FFN-->>Attention: Transformed output
            deactivate FFN
            Attention->>Attention: Add and Norm Residual Connection
        end
        
        Attention-->>Model: Final layer output
        deactivate Attention
        
        Model->>Decoder: Project to vocabulary space
        activate Decoder
        Decoder->>Decoder: Linear hidden_dim to vocab_size
        Decoder->>Decoder: Greedy decoding argmax
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

## 5. Evaluation Flow

This diagram shows the complete evaluation pipeline using the Spider dataset.

```mermaid
sequenceDiagram
    actor User
    participant Script as run_eval.py / run_improved_eval.py
    participant DataLoader as dataset_loader.py
    participant Loader as model_loader.py
    participant Inference as inference.py
    participant Evaluator as evaluate.py
    participant Database as SQLite DB
    
    User->>Script: python run_eval.py --split dev --limit 100
    activate Script
    
    Script->>DataLoader: load_spider_dataset split=dev limit=100
    activate DataLoader
    DataLoader->>DataLoader: Load spider dev.json
    DataLoader->>DataLoader: Parse into SpiderExample objects
    DataLoader-->>Script: List of SpiderExample
    deactivate DataLoader
    
    Script->>Loader: load_model
    activate Loader
    Loader-->>Script: model and tokenizer
    deactivate Loader
    
    loop For each SpiderExample
        Script->>DataLoader: get_database_schema db_id
        activate DataLoader
        DataLoader->>DataLoader: Load tables.json
        DataLoader->>DataLoader: Build CREATE TABLE statements
        DataLoader-->>Script: schema_string
        deactivate DataLoader
        
        Script->>Inference: generate_sql model tokenizer question schema
        activate Inference
        Inference-->>Script: predicted_sql
        deactivate Inference
        
        Script->>Evaluator: compute_exact_match predicted reference
        activate Evaluator
        Evaluator->>Evaluator: normalize_sql predicted
        Evaluator->>Evaluator: normalize_sql reference
        Evaluator->>Evaluator: Compare normalized strings
        Evaluator-->>Script: exact_match bool
        deactivate Evaluator
        
        alt Database path exists
            Script->>Evaluator: compute_execution_accuracy predicted reference db_path
            activate Evaluator
            
            Evaluator->>Database: execute_sql predicted_sql
            activate Database
            Database-->>Evaluator: pred_results
            deactivate Database
            
            Evaluator->>Database: execute_sql reference_sql
            activate Database
            Database-->>Evaluator: ref_results
            deactivate Database
            
            Evaluator->>Evaluator: compare_results pred ref
            Evaluator-->>Script: execution_match bool
            deactivate Evaluator
        else No database
            Script->>Script: execution_match equals False
        end
        
        Script->>Script: Aggregate scores
    end
    
    Script->>Evaluator: compute_metrics exact_matches execution_matches
    activate Evaluator
    Evaluator-->>Script: metrics dict
    deactivate Evaluator
    
    Script-->>User: Report metrics
    deactivate Script
```

## 6. Few-Shot Inference Flow

This diagram shows the enhanced few-shot prompting approach.

```mermaid
sequenceDiagram
    participant Caller as Caller
    participant FewShot as inference_fewshot.py
    participant Tokenizer as PreTrainedTokenizer
    participant Model as Qwen2.5-Coder
    
    Caller->>FewShot: generate_sql_fewshot model tokenizer query schema
    activate FewShot
    
    FewShot->>FewShot: Load FEW_SHOT_EXAMPLES 3 examples
    
    Note over FewShot: Build prompt with examples
    
    FewShot->>FewShot: Construct full prompt with examples plus target query
    
    FewShot->>Tokenizer: tokenizer prompt return_tensors pt
    activate Tokenizer
    Tokenizer-->>FewShot: inputs
    deactivate Tokenizer
    
    FewShot->>Model: model.generate max_new_tokens=256 do_sample=False
    activate Model
    Model-->>FewShot: outputs
    deactivate Model
    
    FewShot->>Tokenizer: tokenizer.decode outputs
    activate Tokenizer
    Tokenizer-->>FewShot: response_text
    deactivate Tokenizer
    
    FewShot->>FewShot: Extract SQL after last SQL marker
    FewShot->>FewShot: Remove markdown formatting
    FewShot->>FewShot: Remove explanatory text
    FewShot->>FewShot: Keep first statement only
    
    FewShot-->>Caller: cleaned_sql
    deactivate FewShot
    FewShot-->>Caller: cleaned_sql
    deactivate FewShot
```

## 7. Improved Inference Flow (Fine-tuned)

This diagram shows the inference flow using the fine-tuned adapter.

```mermaid
sequenceDiagram
    participant Script as run_improved_eval.py
    participant Loader as model_loader.py
    participant Inference as inference_improved.py
    participant Model as Qwen2.5 + LoRA Adapter
    
    Script->>Loader: load_model(adapter_path="...")
    activate Loader
    Loader->>Loader: Load base model
    Loader->>Loader: Load PeftModel with adapter
    Loader-->>Script: model (wrapped)
    deactivate Loader
    
    Script->>Inference: generate_sql_improved(model, ...)
    activate Inference
    Inference->>Model: generate()
    activate Model
    Model->>Model: Use adapter weights for generation
    Model-->>Inference: output tokens
    deactivate Model
    
    Inference->>Inference: Enhanced post-processing
    Inference-->>Script: SQL query
    deactivate Inference
```

## 7. Fine-tuning Flow

This diagram shows the process of fine-tuning the model on the Spider dataset using QLoRA.

```mermaid
sequenceDiagram
    actor User
    participant Script as run_finetuning.py
    participant Trainer as train.py
    participant Loader as dataset_loader.py
    participant Formatter as dataset_formatter.py
    participant HF as HuggingFace
    participant Model as Qwen2.5-Coder
    
    User->>Script: python run_finetuning.py --epochs 3
    activate Script
    
    Script->>Trainer: train()
    activate Trainer
    
    Trainer->>Loader: load_spider_dataset split=train
    activate Loader
    Loader-->>Trainer: train_examples
    deactivate Loader
    
    Trainer->>Formatter: prepare_training_data train_examples
    activate Formatter
    loop For each example
        Formatter->>Formatter: Format into Chat ML
        Formatter->>Formatter: System + User (Schema+Question) + Assistant (SQL)
    end
    Formatter-->>Trainer: formatted_dataset
    deactivate Formatter
    
    Trainer->>HF: AutoModelForCausalLM.from_pretrained
    activate HF
    HF-->>Trainer: model (4-bit quantized)
    deactivate HF
    
    Trainer->>Trainer: Prepare for k-bit training
    Trainer->>Trainer: Apply LoRA config (PEFT)
    
    Trainer->>Trainer: Initialize SFTTrainer
    
    Trainer->>Model: Train loop
    activate Model
    loop For each batch
        Model->>Model: Forward pass
        Model->>Model: Compute Loss
        Model->>Model: Backward pass (update LoRA weights)
    end
    Model-->>Trainer: Training complete
    deactivate Model
    
    Trainer->>Trainer: Save adapter weights
    Trainer-->>Script: Done
    deactivate Trainer
    
    Script-->>User: Fine-tuning complete
    deactivate Script
```

## 8. Dataset Loading Flow

This diagram shows how the Spider dataset is loaded and processed.

```mermaid
sequenceDiagram
    participant Caller as Caller
    participant Loader as dataset_loader.py
    participant FS as File System
    
    Caller->>Loader: load_spider_dataset split=dev limit=None
    activate Loader
    
    Loader->>Loader: Determine spider_dir path
    
    Loader->>FS: Check if spider_dir exists
    activate FS
    alt Directory not found
        FS-->>Loader: False
        Loader-->>Caller: FileNotFoundError
    else Directory exists
        FS-->>Loader: True
        deactivate FS
        
        Loader->>FS: Open spider dev.json
        activate FS
        FS-->>Loader: JSON data
        deactivate FS
        
        loop For each item in JSON
            Loader->>Loader: Create SpiderExample question query db_id db_path
        end
        
        Loader-->>Caller: List of SpiderExample
    end
    deactivate Loader
    
    Note over Caller,Loader: Schema retrieval for evaluation
    
    Caller->>Loader: get_database_schema db_id
    activate Loader
    
    Loader->>Loader: load_spider_tables
    Loader->>FS: Open spider tables.json
    activate FS
    FS-->>Loader: tables_data
    deactivate FS
    
    Loader->>Loader: Find db_id in tables
    Loader->>Loader: Extract table_names column_names column_types
    
    loop For each table
        Loader->>Loader: Build CREATE TABLE statement
    end
    
    Loader-->>Caller: schema_string
    deactivate Loader
```

## 9. Error Handling Flow

This diagram shows error scenarios and how they are handled.

```mermaid
sequenceDiagram
    actor User
    participant Main as run_inference.py
    participant Loader as model_loader.py
    participant HF as HuggingFace
    
    User->>Main: Start application
    Main->>Loader: load_model
    activate Loader
    
    Loader->>HF: AutoTokenizer.from_pretrained
    activate HF
    
    alt Model Not Found
        HF-->>Loader: HTTPError ModelNotFoundError
        Loader->>Loader: Catch Exception
        Loader->>Loader: Print error message
        Loader-->>Main: Raise Exception
        Main-->>User: Error Model loading failed
    else Network Error
        HF-->>Loader: ConnectionError
        Loader->>Loader: Catch Exception
        Loader-->>Main: Raise Exception
        Main-->>User: Error Check internet connection
    else Out of Memory
        HF-->>Loader: CUDA OOM MemoryError
        Note over Loader: Falls back to CPU with float32
        Loader-->>Main: model on CPU and tokenizer
        Main-->>User: Warning Running on CPU
    else Success
        HF-->>Loader: tokenizer
        deactivate HF
        Loader-->>Main: model and tokenizer
        Main-->>User: Success
    end
    deactivate Loader
```

## 10. SQL Normalization and Comparison Flow

This diagram shows how SQL queries are normalized for exact match comparison.

```mermaid
sequenceDiagram
    participant Eval as evaluate.py
    participant Norm as normalize_sql
    
    Eval->>Eval: compute_exact_match predicted reference
    activate Eval
    
    Eval->>Norm: normalize_sql predicted
    activate Norm
    Norm->>Norm: Convert to lowercase
    Norm->>Norm: Remove extra whitespace with regex
    Norm->>Norm: Remove trailing semicolon
    Norm->>Norm: Strip leading and trailing whitespace
    Norm-->>Eval: normalized_predicted
    deactivate Norm
    
    Eval->>Norm: normalize_sql reference
    activate Norm
    Norm->>Norm: Same normalization steps
    Norm-->>Eval: normalized_reference
    deactivate Norm
    
    Eval->>Eval: Compare normalized_predicted equals normalized_reference
    Eval-->>Eval: Return True or False
    deactivate Eval
```

---

## Component Interaction Summary

### Key Components

1. **run_eval.py** - Entry point for baseline evaluation
2. **run_improved_eval.py** - Entry point for fine-tuned evaluation
3. **rag.py** - Retrieves relevant database schemas using semantic search
4. **model_loader.py** - Downloads and initializes the Qwen2.5-Coder-0.5B-Instruct model
5. **inference.py** - Zero-shot SQL generation logic
6. **inference_fewshot.py** - Few-shot SQL generation with example prompts
7. **dataset_loader.py** - Loads Spider dataset and database schemas
8. **evaluate.py** - Metrics computation exact match and execution accuracy

### Data Flow

```
User Query 
  → RAG Retrieval find relevant schema 
  → Prompt Construction Instruction plus Schema plus Question 
  → Tokenization text to token IDs 
  → Model Inference Qwen2.5-Coder transformer 
  → Detokenization token IDs to text 
  → Post-processing extract SQL clean formatting 
  → Generated SQL
```

### Evaluation Flow

```
Spider Dataset dev.json
  → Load Examples question reference SQL db_id
  → For each example
      → Get schema from tables.json
      → Generate SQL using model
      → Compute Exact Match string comparison
      → Compute Execution Accuracy run on SQLite compare results
  → Aggregate metrics
  → Report Exact Match percent Execution Accuracy percent
```

### Model Specifications

| Property | Value |
|----------|-------|
| Model | Qwen2.5-Coder-0.5B-Instruct |
| Parameters | 500M |
| Context Length | 32768 tokens |
| Embedding Model | all-MiniLM-L6-v2 384-dim |
| Inference | Greedy decoding do_sample=False |
