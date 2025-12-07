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
    prompt = f"### Instruction:\n{query}\n\n### Context:\n{schema_context}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Basic post-processing to extract SQL if needed (depending on model output format)
    # For now, returning the full response or a substring after "Response:"
    if "### Response:" in response:
        return response.split("### Response:")[-1].strip()
    return response.strip()
