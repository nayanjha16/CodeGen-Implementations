from fastapi import FastAPI, HTTPException
from .schemas import QueryRequest, QueryResponse
from .services.orchestrator import Orchestrator

app = FastAPI(title="Cognitive Bridge API")
orchestrator = Orchestrator()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Cognitive Bridge Backend is running"}

@app.post("/api/v1/generate", response_model=QueryResponse)
async def generate_response(request: QueryRequest):
    try:
        response = await orchestrator.generate(request.db_id, request.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
