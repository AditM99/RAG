Graph-Powered Conversational Search (RAG + Neo4j Vector DB)
Example project demonstrating a Hybrid RAG flow using Neo4j as both a knowledge graph and a vector database.
Use Hugging Face embeddings to generate passage vectors and store them directly in Neo4j.
Use spaCy to extract entities and store them as nodes/relationships in Neo4j (entity graph).

Hybrid retrieval:
Neo4j vector index returns relevant passages.
Neo4j graph returns connected entities.
Both are merged into a prompt for the LLM.

FastAPI backend exposes /ingest and /query endpoints for ingestion and question answering.

What is included

backend/ : FastAPI app, ingestion and retrieval code
docker-compose.yml : spins up Neo4j for local testing
.env.template : environment variables required
requirements.txt : Python packages
example_data/ : small example text file to ingest

Setup (quick)
Copy .env.template to .env and fill values:
HUGGINGFACEHUB_API_TOKEN (required for Hugging Face models)
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (if using local docker-compose, defaults are in the template)

Create a Python venv and install requirements:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Start Neo4j locally for testing:

docker-compose up -d

# open http://localhost:7474 (browser) and login with neo4j/letmein (see .env.template)

Create a vector index in Neo4j (one-time setup, example for 384-dim embeddings):

CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (c:Passage) ON (c.embedding)
OPTIONS {indexConfig: {
`vector.dimensions`: 384,
`vector.similarity_function`: 'cosine'
}};

Run the API:

uvicorn backend.main:app --reload --port 8000

Ingest a file (example provided):
curl -X POST "http://localhost:8000/ingest" -F "file=@example_data/example.txt"

Query:
curl -X POST "http://localhost:8000/query" \
 -H "Content-Type: application/json" \
 -d '{"query":"What unusual activity was flagged on John Doe’s account?"}'

Notes & caveats

All semantic search is performed inside Neo4j’s vector index, no external vector DB like Pinecone is needed.
In production, you should add batching, error handling, authentication, rate-limiting, and robust entity linking.
The LLM used here is Hugging Face (flan-t5-base or similar). You can swap it for OpenAI or any other provider.
Keep an eye on embedding dimensions — they must match between Hugging Face model and Neo4j vector index configuration.
