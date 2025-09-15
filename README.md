# Graph-Powered Conversational Search (RAG + Graph)
Example project demonstrating a Hybrid RAG flow:
- Use OpenAI embeddings + Pinecone for semantic passage retrieval
- Use spaCy to extract entities and store in Neo4j (entity graph)
- Hybrid retrieval: Pinecone returns relevant passages; Neo4j returns connected entities; both are merged into a prompt for the LLM
- FastAPI backend exposes `/ingest` and `/query` endpoints

## What is included
- `backend/` : FastAPI app, ingestion and retrieval code
- `docker-compose.yml` : spins up Neo4j for local testing (Pinecone is a hosted service)
- `.env.template` : environment variables required
- `requirements.txt` : Python packages
- `example_data/` : small example text file to ingest

## Setup (quick)
1. Copy `.env.template` to `.env` and fill values:
   - `OPENAI_API_KEY` (required)
   - `PINECONE_API_KEY` and `PINECONE_ENV` (optional if you intend to use Pinecone)
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (if using local docker-compose, defaults are in the template)
2. Create a Python venv and install requirements:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. (Optional) Start Neo4j locally for testing:
```bash
docker-compose up -d
# open http://localhost:7474 (browser) and login with neo4j/letmein (see .env.template)
```
4. Run the API:
```bash
uvicorn backend.main:app --reload --port 8000
```
5. Ingest a file (example provided):
```bash
curl -X POST "http://localhost:8000/ingest" -F "file=@example_data/example.txt"
```
6. Query:
```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query":"how do I reset my password for my account?"}'
```

## Notes & caveats
- Pinecone is a hosted vector DB — you must create an index and supply credentials. The example includes code to create an index if missing.
- This example is intentionally simple for clarity. In production you should add batching, error handling, authentication, rate-limiting, and robust entity linking.
- The OpenAI usage in this project uses `text-davinci-003` style calls via `openai.ChatCompletion` or `OpenAI` in LangChain. Update to the model of your choice and follow the provider's best practices for system prompts and token limits.
