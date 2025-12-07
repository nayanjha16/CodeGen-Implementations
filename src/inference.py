from transformers import PreTrainedModel, PreTrainedTokenizer

def generate_sql(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    query: str, 
    schema_context: str
) -> str:
    """
    Generates SQL from a natural language query and schema context.
    
    Args:
        model: The pre-trained language model
        tokenizer: The tokenizer for the model
        query: Natural language query from user
        schema_context: Database schema information
        
    Returns:
        Generated SQL query as a string
    """
    prompt = f"""### Instruction:
You are a text-to-SQL generator. Given the database schema and a natural language question, generate a valid SQL query.
Return ONLY the SQL query, without any explanation or markdown formatting.

### Schema:
{schema_context}

### Question:
{query}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
        
    # Clean up response
    # 1. Remove markdown code blocks
    if "```sql" in response:
        response = response.split("```sql")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
        
    # 2. Remove any leading/trailing whitespace or quotes
    response = response.strip().strip('"').strip("'")
    
    # 3. Keep only the first statement if multiple are generated (simple heuristic)
    if ";" in response:
        response = response.split(";")[0] + ";"
        
    return response
