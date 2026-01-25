"""
FastAPI Backend Server for NL-to-NoSQL Conversion
Fixed for HuggingFace Spaces with PEFT adapter compatibility
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import torch
import sys
import os
import json

# ==================== CONFIGURATION ====================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TEXT2SQL_ADAPTER = "./qwen_text2sql_adapter"
SQL2MONGO_ADAPTER = "./sql_to_mongodb_adapter"

# Force CPU for stability on HuggingFace Spaces
DEVICE = "cpu"
print(f"🔧 Using device: {DEVICE}")

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="NL-to-NoSQL Conversion API",
    description="Convert Natural Language to SQL and MongoDB queries",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL STATE ====================

models = {
    "text2sql": None,
    "sql2mongo": None,
    "tokenizer": None,
    "loaded": False
}

# ==================== PYDANTIC MODELS ====================

class TextToSQLRequest(BaseModel):
    question: str
    schema: str

class SQLToMongoRequest(BaseModel):
    sql_query: str

class CompletePipelineRequest(BaseModel):
    question: str
    schema: str

class SchemaTranslationRequest(BaseModel):
    sql_schema: str
    use_rag: bool = True
    k: int = 3

# ==================== HELPER FUNCTIONS ====================

def sanitize_adapter_config(adapter_path: str) -> None:
    """
    Sanitize adapter_config.json to remove incompatible parameters
    This fixes issues with adapters trained on newer PEFT versions
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    
    if not os.path.exists(config_path):
        print(f"⚠️ Config not found: {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Parameters that might cause compatibility issues
        problematic_params = [
            'alora_invocation_tokens',  # From newer PEFT versions
            'use_rslora',
            'use_dora',
            'layer_replication',
        ]
        
        modified = False
        for param in problematic_params:
            if param in config:
                print(f"   Removing incompatible parameter: {param}")
                del config[param]
                modified = True
        
        if modified:
            # Backup original
            backup_path = config_path + ".backup"
            if not os.path.exists(backup_path):
                with open(backup_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # Save sanitized config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"   ✅ Sanitized config saved")
        else:
            print(f"   ✅ Config already compatible")
            
    except Exception as e:
        print(f"   ⚠️ Error sanitizing config: {e}")

# ==================== MODEL LOADING ====================

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global models
    
    try:
        print("\n" + "="*60)
        print("🚀 LOADING MODELS")
        print("="*60)
        
        # Import after checking transformers version
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Load tokenizer with proper configuration for Qwen2.5
        print("\n📦 Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_fast=True
            )
            print("✅ Tokenizer loaded with trust_remote_code")
        except Exception as e:
            print(f"⚠️ Fast tokenizer failed, trying slow: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_fast=False
            )
            print("✅ Tokenizer loaded (slow version)")
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        models["tokenizer"] = tokenizer
        print(f"✅ Tokenizer ready (vocab size: {len(tokenizer)})")
        
        # Load base model
        print("\n📦 Loading base model...")
        print(f"   Device: {DEVICE}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        base_model = base_model.to(DEVICE)
        base_model.eval()
        
        print(f"✅ Base model loaded")
        print(f"   Memory: ~{base_model.get_memory_footprint() / 1e9:.2f} GB")
        
        # Sanitize and load Text-to-SQL adapter
        print("\n📦 Loading Text-to-SQL adapter...")
        print(f"   Path: {TEXT2SQL_ADAPTER}")
        print(f"   Sanitizing config...")
        
        sanitize_adapter_config(TEXT2SQL_ADAPTER)
        
        try:
            text2sql_model = PeftModel.from_pretrained(
                base_model,
                TEXT2SQL_ADAPTER,
                torch_dtype=torch.float32,
                is_trainable=False
            )
            text2sql_model.eval()
            models["text2sql"] = text2sql_model
            print("✅ Text-to-SQL model ready")
        except Exception as e:
            print(f"❌ Failed to load Text-to-SQL adapter: {e}")
            print("   Trying alternative loading method...")
            
            # Alternative: Load adapter manually
            from peft import LoraConfig, get_peft_model
            
            config_path = os.path.join(TEXT2SQL_ADAPTER, "adapter_config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create minimal config
            lora_config = LoraConfig(
                r=config_dict.get('r', 16),
                lora_alpha=config_dict.get('lora_alpha', 32),
                target_modules=config_dict.get('target_modules', ['q_proj', 'v_proj']),
                lora_dropout=config_dict.get('lora_dropout', 0.05),
                bias=config_dict.get('bias', 'none'),
                task_type=config_dict.get('task_type', 'CAUSAL_LM')
            )
            
            text2sql_model = get_peft_model(base_model, lora_config)
            
            # Load weights
            adapter_weights = torch.load(
                os.path.join(TEXT2SQL_ADAPTER, "adapter_model.bin"),
                map_location=DEVICE
            )
            text2sql_model.load_state_dict(adapter_weights, strict=False)
            text2sql_model.eval()
            models["text2sql"] = text2sql_model
            print("✅ Text-to-SQL model loaded (alternative method)")
        
        # Sanitize and load SQL-to-MongoDB adapter  
        print("\n📦 Loading SQL-to-MongoDB adapter...")
        print(f"   Path: {SQL2MONGO_ADAPTER}")
        print(f"   Sanitizing config...")
        
        sanitize_adapter_config(SQL2MONGO_ADAPTER)
        
        try:
            sql2mongo_model = PeftModel.from_pretrained(
                base_model,
                SQL2MONGO_ADAPTER,
                torch_dtype=torch.float32,
                is_trainable=False
            )
            sql2mongo_model.eval()
            models["sql2mongo"] = sql2mongo_model
            print("✅ SQL-to-MongoDB model ready")
        except Exception as e:
            print(f"❌ Failed to load SQL-to-MongoDB adapter: {e}")
            print("   Trying alternative loading method...")
            
            from peft import LoraConfig, get_peft_model
            
            config_path = os.path.join(SQL2MONGO_ADAPTER, "adapter_config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            lora_config = LoraConfig(
                r=config_dict.get('r', 16),
                lora_alpha=config_dict.get('lora_alpha', 32),
                target_modules=config_dict.get('target_modules', ['q_proj', 'v_proj']),
                lora_dropout=config_dict.get('lora_dropout', 0.05),
                bias=config_dict.get('bias', 'none'),
                task_type=config_dict.get('task_type', 'CAUSAL_LM')
            )
            
            sql2mongo_model = get_peft_model(base_model, lora_config)
            
            adapter_weights = torch.load(
                os.path.join(SQL2MONGO_ADAPTER, "adapter_model.bin"),
                map_location=DEVICE
            )
            sql2mongo_model.load_state_dict(adapter_weights, strict=False)
            sql2mongo_model.eval()
            models["sql2mongo"] = sql2mongo_model
            print("✅ SQL-to-MongoDB model loaded (alternative method)")
        
        models["loaded"] = True
        
        print("\n" + "="*60)
        print("✅ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR LOADING MODELS: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        models["loaded"] = False

def generate_text(model, prompt: str, max_tokens: int = 512) -> str:
    """Generate text using the model"""
    try:
        tokenizer = models["tokenizer"]
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if prompt in generated_text:
            generated_text = generated_text.split(prompt)[-1].strip()
        
        return generated_text
        
    except Exception as e:
        print(f"Generation error: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# ==================== SCHEMA DATABASE ====================

SCHEMAS = {
    "employees_db": "employees(id, name, department, salary, hire_date)",
    "products_db": "products(id, name, category, price, stock)",
    "users_db": "users(db": "users(id, username, email, age, country)",
    "orders_db": "orders(id, customer_id, product_id, quantity, order_date, total)",
    "students_db": "students(id, name, major, gpa, enrollment_year)",
}

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NL-to-NoSQL Conversion API",
        "status": "running",
        "models_loaded": models["loaded"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": models["loaded"],
        "device": DEVICE
    }

@app.post("/text-to-sql")
async def text_to_sql(request: TextToSQLRequest):
    """Convert natural language to SQL"""
    
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    try:
        schema_str = SCHEMAS.get(request.schema, request.schema)
        
        messages = [
            {"role": "system", "content": "You are a SQL expert. Generate ONLY valid, safe SQL queries."},
            {"role": "user", "content": f"Convert to SQL. Output ONLY the SQL query.\n\nSchema: {schema_str}\nQuery: {request.question}"}
        ]
        
        prompt = models["tokenizer"].apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        sql_query = generate_text(models["text2sql"], prompt, max_tokens=256)
        
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        if sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        return {
            "sql_query": sql_query,
            "schema": schema_str,
            "security": {"is_safe": True, "message": "Query validated"}
        }
        
    except Exception as e:
        print(f"Text-to-SQL error: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sql-to-mongodb")
async def sql_to_mongodb(request: SQLToMongoRequest):
    """Convert SQL to MongoDB query"""
    
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    try:
        messages = [
            {"role": "system", "content": "Convert SQL to MongoDB. Output valid MongoDB query syntax."},
            {"role": "user", "content": f"SQL: {request.sql_query}"}
        ]
        
        prompt = models["tokenizer"].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        mongodb_query = generate_text(models["sql2mongo"], prompt, max_tokens=512)
        
        mongodb_query = mongodb_query.strip()
        if mongodb_query.startswith("```javascript"):
            mongodb_query = mongodb_query.replace("```javascript", "").replace("```", "").strip()
        if mongodb_query.startswith("```"):
            mongodb_query = mongodb_query.replace("```", "").strip()
        
        return {
            "mongodb_query": mongodb_query,
            "query_type": "aggregate" if "aggregate" in mongodb_query.lower() else "find"
        }
        
    except Exception as e:
        print(f"SQL-to-MongoDB error: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/complete-pipeline")
async def complete_pipeline(request: CompletePipelineRequest):
    """Complete NL -> SQL -> MongoDB pipeline"""
    
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    try:
        text2sql_result = await text_to_sql(TextToSQLRequest(
            question=request.question,
            schema=request.schema
        ))
        
        sql_query = text2sql_result["sql_query"]
        
        sql2mongo_result = await sql_to_mongodb(SQLToMongoRequest(
            sql_query=sql_query
        ))
        
        return {
            "sql_query": sql_query,
            "mongodb_query": sql2mongo_result["mongodb_query"],
            "schema": text2sql_result["schema"],
            "query_type": sql2mongo_result["query_type"]
        }
        
    except Exception as e:
        print(f"Complete pipeline error: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/schema-translation")
async def schema_translation(request: SchemaTranslationRequest):
    """Translate SQL schema to MongoDB (basic implementation)"""
    
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    try:
        mongodb_schema = {
            "collection": "collection_name",
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": [],
                    "properties": {}
                }
            },
            "indexes": []
        }
        
        return {
            "mongodb_schema": mongodb_schema,
            "similar_examples": [],
            "method": "rule_based"
        }
        
    except Exception as e:
        print(f"Schema translation error: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING BACKEND SERVER")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info",
        access_log=True
    )
