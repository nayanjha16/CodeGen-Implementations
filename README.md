The project explores multiple generations of approaches, starting from classical NLP and rule-based systems, progressing through transformer fine-tuning, and finally converging on LLM-based reasoning with Retrieval-Augmented Generation (RAG) and execution feedback.

The emphasis is not just on syntactic translation, but on execution correctness, making this project closer to real-world database systems rather than academic-only benchmarks.

🎯 Key Objectives
Translate natural language questions into executable SQL
Convert SQL queries into MongoDB aggregation pipelines
Validate correctness via execution-based evaluation
Compare traditional, neural, and LLM-based approaches
Study why certain approaches fail and how LLMs + RAG improve robustness

📊 Datasets Used
1️⃣ Spider (Text-to-SQL)
  Cross-database, cross-domain benchmark
  200+ databases with diverse schemas
  Complex SQL including:
    JOINs
    Nested queries
    GROUP BY / HAVING
    Subqueries
    Industry-standard dataset used by:
  RAT-SQL
  T5-based systems

2️⃣ BirdBench (SQL-to-NoSQL)
    Execution-grounded benchmark
    Provides ground-truth SQL
    SQLite databases for execution
Ideal for validating SQL → MongoDB semantic equivalence
Used for full execution-loop evaluation

🧩 Part 1 — Text-to-SQL Models
🔹 Model 1: SQLNet (LSTM Baseline)
    Motivation
      Establish a classical NLP baseline
      Understand limitations of sequence-to-sequence LSTMs
    Approach
      Encoder–decoder LSTM
      Token-based SQL generation
      No schema linking
    Outcome
      Near-zero execution accuracy
      Failed on multi-table queries
      Demonstrated why LSTMs are insufficient for realistic SQL
🔹 Model 2: T5-Base Fine-Tuned on Spider
    Motivation
      Leverage transformer encoder–decoder architecture
      Handle SQL as a generation task
    Approach
      Input: Natural language + schema serialized as text
      Output: SQL query
      Fine-tuned on Spider dataset
    Results
      Exact Match (EM): ~24%
      Execution Accuracy (EX): ~37%
    Limitations
      Weak schema grounding
      SQL correctness does not always imply execution correctness
🔹 Model 3: RAT-SQL-Lite + T5 (Graph-Aware)
    Motivation
      Inject schema structure explicitly
      Model relational reasoning
    Approach
      Schema graph:
      Tables
      Columns
      PK–FK relationships
      Graph serialized and fed to T5
      Hybrid symbolic + neural reasoning
    Results
      EM ≈ 28.5%
      EX ≈ 42% (best Text-to-SQL model)
🔹 Model 4: Pretrained LLM (Benchmark)
    Motivation
      Compare against modern pretrained LLMs
      No training or fine-tuning
    Outcome
      Strong execution accuracy
      Lower exact match
      Motivated LLM-based solutions for SQL reasoning

🔄 Part 2 — SQL-to-NoSQL (MongoDB) Translation
🔹 Model 1: Rule-Based SQL → MongoDB (Compiler Style)
    Motivation
      Create a deterministic baseline
      Understand exact SQL ↔ MongoDB operator mapping
    Approach
      SQL parsed into AST
      Handcrafted rules:
        SELECT → $project
        WHERE → $match
        JOIN → $lookup
        GROUP BY → $group
    Outcome
      Worked only for simple SQL
    Failed on:
      Nested queries
      Complex aggregations
      Advanced JOIN patterns
      Not scalable
🔹 Model 2: T5 Fine-Tuned SQL → MongoDB
    Motivation
      Learn SQL → MongoDB translation automatically
    Approach
      Generated training data using rule-based model
      Fine-tuned T5 encoder–decoder
    Outcome
      Low training loss but poor execution accuracy
      Model learned to generate empty or trivial pipelines
      Highlighted weakness of synthetic supervision


🚀 Part 3 — LLM-Based SQL-to-NoSQL (Core Contribution)
🔹 Model 3A: LLM Baseline (Prompt-Based)
    Motivation
      Leverage LLM reasoning instead of brittle rules
      Avoid fine-tuning
    Approach
      Prompt-based SQL → MongoDB conversion
      No schema awareness
    Outcome
      Better coverage than rule-based systems
      Frequent failures due to missing schema grounding
🔹 Model 3B: LLM + RAG (Schema-Aware)
    RAG v1 — Schema Injection
      Tables and columns injected into prompt
      Improved grounding, limited JOIN reasoning
    RAG v2 — Foreign-Key Awareness
      PK–FK relationships retrieved from SQLite
      Better JOIN handling
    RAG v3 — Join Template Injection
      Explicit MongoDB $lookup templates constructed
      LLM guided using executable join patterns
    Execution feedback loop added
    Evaluation
      SQL executed on SQLite
      MongoDB pipeline executed locally
      Results normalized and compared
      Success defined by execution equivalence
    Outcome
      ~30% execution accuracy on BirdBench Mini-Dev
      Significant improvement without fine-tuning

🔁 Execution Feedback Loop (Key Innovation)
    Execute SQL on SQLite
    Generate MongoDB pipeline via LLM
    Execute pipeline on MongoDB
    Compare results
    If mismatch:
      Feed error + context back to LLM
      Retry correction
      Log retries and final outcome
  This makes the system self-correcting.


📈 Key Learnings
  Rule-based systems do not scale
  Fine-tuning on weak data fails silently
  Schema grounding is essential
  Execution-based evaluation is critical

LLMs + RAG outperform traditional pipelines without training



🏁 Final Note
  This project demonstrates a full-stack AI system combining:
    NLP
    Transformers
    Graph reasoning
    Database systems
    LLM orchestration
    Execution-based evaluation
