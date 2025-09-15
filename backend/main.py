import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "letmein123")

HF_TOKEN = os.getenv("HF_TOKEN")
print(f"HF_TOKEN exists: {bool(HF_TOKEN)}")
print(f"HF_TOKEN length: {len(HF_TOKEN) if HF_TOKEN else 0}")

if HF_TOKEN:
    print(f"HF_TOKEN starts with: {HF_TOKEN[:10]}...")
else:
    print("HF_TOKEN not found in environment variables")
    print("Available env vars:", [k for k in os.environ.keys() if 'HF' in k or 'HUGGING' in k])

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


from backend.ingest import ingest_file
from backend.rag import answer_query

class QueryRequest(BaseModel):
    query: str

app = FastAPI(
    title="Graph-Powered RAG Backend",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    try:
        ingest_file(text, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"status": "ok", "filename": file.filename})

@app.post("/query")
def query(request: QueryRequest):
    try:
        # Load token here and pass it to the function
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        result = answer_query(request.query, hf_token)
        return result
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)
