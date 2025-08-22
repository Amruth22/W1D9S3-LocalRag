import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CACHE_DIR = DATA_DIR / "cache"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# Model configurations
LLM_MODEL_ID = "meta-llama/Llama-3.2-1B"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"
HF_TOKEN = "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"  # You should use environment variables for this

# FAISS configuration
FAISS_INDEX_FILE = EMBEDDINGS_DIR / "faiss_index.bin"
FAISS_METADATA_FILE = EMBEDDINGS_DIR / "metadata.json"

# Cache configuration
CACHE_TTL = 24 * 60 * 60  # 24 hours in seconds
CACHE_MAX_SIZE = 1000  # Maximum number of items in cache

# RAG configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

# LLM configuration
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 50