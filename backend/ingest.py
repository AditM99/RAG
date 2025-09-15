import os, uuid
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
import spacy
import numpy as np


NEO4J_URI = os.getenv('NEO4J_URI','bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER','neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD','letmein123')

nlp = spacy.load('en_core_web_sm')
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def init_neo4j():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver

def extract_entities(text: str):
    doc = nlp(text)
    # return unique entity texts
    return list({ent.text.strip() for ent in doc.ents})


def ingest_file(text: str, filename: str = None):
    passages = chunk_text(text)
    ids = [str(uuid.uuid4()) for _ in passages]
    embeddings = embed_model.embed_documents(passages)  # list of lists

    driver = init_neo4j()
    with driver.session() as session:
        # Store passages as nodes with embeddings
        for i, passage in enumerate(passages):
            session.run("""
                MERGE (p:Passage {id:$id})
                SET p.text = $text,
                    p.filename = $filename,
                    p.embedding = $embedding
            """, id=ids[i], text=passage, filename=filename or "unknown", embedding=embeddings[i])

        # Extract entities and create nodes
        entities = extract_entities(text)
        for ent in entities:
            session.run("MERGE (e:Entity {name:$name})", name=ent)
        # Connect every pair of entities in this document
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                session.run("""
                    MATCH (a:Entity {name:$a}), (b:Entity {name:$b})
                    MERGE (a)-[r:RELATED_TO]->(b)
                    ON CREATE SET r.count = 1
                    ON MATCH SET r.count = r.count + 1
                """, a=entities[i], b=entities[j])
    driver.close()