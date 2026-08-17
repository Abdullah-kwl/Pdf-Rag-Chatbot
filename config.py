"""Hardcoded configuration for the RAG chatbot."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DB_DIR = BASE_DIR / "db"

# Embeddings
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cpu"
NORMALISE_EMBEDDINGS = True

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Retrieval
COLLECTION_NAME = "rag_chunks"
RETRIEVER_SEARCH_TYPE = "similarity_score_threshold"
RETRIEVER_K = 5
SCORE_THRESHOLD = 0.3

# Generation
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.0
HISTORY_WINDOW = 6
