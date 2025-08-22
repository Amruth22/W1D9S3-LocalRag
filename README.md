# Optimized Local RAG System

This is an optimized Retrieval-Augmented Generation (RAG) system that runs entirely locally using:
- Llama 3.2 1B model for text generation
- all-MiniLM-L6-v2 for embeddings
- FAISS for vector storage

## Features
- Local document processing and indexing
- Semantic search using embeddings
- Dual-layer caching (in-memory and disk-based)
- Optimized for CPU-only execution

## Installation

1. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface
```bash
python main.py
```

## Testing

To run unit tests for the RAG system:
```bash
python unit_test.py
```

## Adding Documents

To add documents for the RAG system to use:
1. Place your text documents in the `data/documents/` directory
2. Run the system (CLI interface)
3. The documents will be automatically processed and indexed

## Troubleshooting

If the RAG system is not finding information from your documents:
1. Make sure your documents are in the `data/documents/` directory
2. Check that the documents were processed by looking for output messages during startup
3. Clear the cache by deleting files in `data/cache/` if needed
4. Delete the FAISS index files (`embeddings/faiss_index.bin` and `embeddings/metadata.json`) to force reprocessing

## Configuration

Configuration settings can be found in `config/settings.py`:
- Model IDs
- Cache settings
- Chunking parameters
- LLM generation parameters