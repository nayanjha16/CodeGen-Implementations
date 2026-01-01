"""
LLM Client Utility
Handles interactions with various LLM providers (OpenAI, Anthropic, local models)
"""
from typing import Dict, List, Optional, Any
import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.0, max_tokens: int = 2000) -> str:
        """Generate completion from prompt"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize OpenAI client
        
        Args:
            model: Model name (gpt-4, gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (if None, reads from env)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI client with model: {model}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 2000) -> str:
        """
        Generate completion using OpenAI API
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", 
                 api_key: Optional[str] = None):
        """
        Initialize Anthropic client
        
        Args:
            model: Model name
            api_key: Anthropic API key (if None, reads from env)
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"Initialized Anthropic client with model: {model}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 2000) -> str:
        """
        Generate completion using Anthropic API
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = self.client.messages.create(**kwargs)
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class LocalLLMClient(BaseLLMClient):
    """Client for locally hosted LLMs (e.g., via transformers)"""
    
    def __init__(self, model_path: str):
        """
        Initialize local LLM client
        
        Args:
            model_path: Path to local model
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers torch")
        
        self.model_path = model_path
        
        logger.info(f"Loading local model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        logger.info("Local model loaded successfully")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 2000) -> str:
        """
        Generate completion using local model
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Combine system and user prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        if generated_text.startswith(full_prompt):
            generated_text = generated_text[len(full_prompt):].strip()
        
        return generated_text


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client"""
    
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client
        
        Args:
            model: Model name (llama3.1:8b, llama2:7b, etc.)
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url
        logger.info(f"Initialized Ollama client with model: {model}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 2000) -> str:
        """
        Generate completion using Ollama
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        import requests
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    @staticmethod
    def create_client(provider: str, model: str, **kwargs) -> BaseLLMClient:
        """
        Create an LLM client based on provider
        
        Args:
            provider: Provider name (openai, anthropic, local, ollama)
            model: Model name or path
            **kwargs: Additional arguments for the client
            
        Returns:
            LLM client instance
        """
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAIClient(model=model, **kwargs)
        elif provider == "anthropic":
            return AnthropicClient(model=model, **kwargs)
        elif provider == "local":
            return LocalLLMClient(model_path=model, **kwargs)
        elif provider == "ollama":
            return OllamaClient(model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class PromptTemplate:
    """Templates for various prompts in the Text-to-NoSQL system"""
    
    @staticmethod
    def schema_prediction_prompt(nlq: str, schemas: str) -> Dict[str, str]:
        """
        Create prompt for schema prediction
        
        Args:
            nlq: Natural language query
            schemas: MongoDB schemas (formatted)
            
        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = (
            "You are the MongoDB natural language interface, responsible for converting "
            "user input natural language queries into MongoDB query statements based on "
            "the MongoDB Collection and their Fields, and parsing the features according "
            "to user requirements."
        )
        
        user = f"""# Given the natural language query, please predict the fields used in the query.
## Natural Language Query: `{nlq}`
## MongoDB Collection and their Fields
{schemas}

Please respond with ONLY the field names separated by commas, nothing else."""
        
        return {"system": system, "user": user}
    
    @staticmethod
    def query_generation_prompt(nlq: str, schemas: str) -> Dict[str, str]:
        """
        Create prompt for query generation
        
        Args:
            nlq: Natural language query
            schemas: MongoDB schemas (formatted)
            
        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = (
            "You are the MongoDB natural language interface, responsible for converting "
            "user input natural language queries into MongoDB query statements based on "
            "the MongoDB collections and their fields."
        )
        
        user = f"""# Generate MongoDB query for the natural language question.

## Question
`{nlq}`

## Available Collections
{schemas}

## Syntax Rules:
1. Use countDocuments() not count()
2. Operators use strings: {{$min: "$age"}} not {{$min: ["$age"]}}
3. Quote field names: {{"field": 1}}
4. Use lowercase null

## Operation Guidance:
- .find() → filtering, projection, sorting
- .distinct() → unique values
- .aggregate() → grouping, calculations across documents

Respond with only the MongoDB query ending with semicolon.
Example: db.collection.find({{}}, {{"field": 1}});"""
        
        return {"system": system, "user": user}
    
    @staticmethod
    def query_refinement_prompt(nlq: str, initial_query: str, 
                               predicted_schemas: str, examples: List[Dict]) -> Dict[str, str]:
        """
        Create prompt for query refinement using RAG
        
        Args:
            nlq: Natural language query
            initial_query: Initially generated query
            predicted_schemas: Predicted schema elements
            examples: Retrieved similar examples
            
        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = (
            "You are an expert MongoDB query optimizer. Refine the given query based on "
            "predicted schemas and similar examples to ensure accuracy and correctness."
        )
        
        examples_text = "\n\n".join([
            f"Example {i+1}:\n"
            f"Question: {ex['question']}\n"
            f"Query: {ex['query']}"
            for i, ex in enumerate(examples)
        ])
        
        user = f"""# Refine the MongoDB query based on similar examples and predicted schemas

## Natural Language Query
`{nlq}`

## Initial Generated Query
```
{initial_query}
```

## Predicted Schemas
{predicted_schemas}

## Similar Examples
{examples_text}

Please provide a refined MongoDB query that:
1. Uses the predicted schemas correctly
2. Follows patterns from similar examples
3. Fixes any potential errors in the initial query
4. Returns accurate results for the given question

Respond with ONLY the refined MongoDB query, nothing else."""
        
        return {"system": system, "user": user}
    
    @staticmethod
    def query_debugging_prompt(nlq: str, query: str, 
                              error: str, schemas: str) -> Dict[str, str]:
        """
        Create prompt for debugging failed queries
        
        Args:
            nlq: Natural language query
            query: Failed query
            error: Error message
            schemas: MongoDB schemas
            
        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = (
            "You are a MongoDB query debugger. Fix the given query that produced an error."
        )
        
        user = f"""# Debug and fix the MongoDB query

## Natural Language Query
`{nlq}`

## Failed Query
```
{query}
```

## Error Message
```
{error}
```

## MongoDB Schemas
{schemas}

## Common Error Fixes:
- "name 'null' is not defined" → Replace unquoted null with None or use "null" as string
- "Unsupported query type" with .count() → Change to .countDocuments()
- "expects a single argument" for $min/$max → Remove array brackets: {{$min: "$field"}} not {{$min: ["$field"]}}
- Syntax errors → Ensure all field names in aggregation are double-quoted
- "name 'X' is not defined" → Add quotes around field name X

Please provide a corrected MongoDB query that:
1. Fixes the specific error mentioned above
2. Uses proper MongoDB syntax (quotes, operators, methods)
3. Returns the expected results for the question

Respond with ONLY the corrected query, nothing else (must end with semicolon)."""
        
        return {"system": system, "user": user}
