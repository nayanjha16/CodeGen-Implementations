from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The natural language query")
    db_id: str = Field(..., description="The target BIRD database ID")

class QueryResponse(BaseModel):
    sql_query: str
    mongo_pipeline: List[Dict[str, Any]]
    sql_result: List[Dict[str, Any]]
    mongo_result: List[Dict[str, Any]]
    execution_match: bool
    explanation: Optional[str] = None
    error: Optional[str] = None

class MigrationRequest(BaseModel):
    db_id: str

class MigrationStatus(BaseModel):
    db_id: str
    status: str
    message: Optional[str] = None
