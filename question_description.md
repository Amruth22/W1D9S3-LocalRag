# RAG System Question Description

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that runs entirely locally. The system combines document retrieval with language model generation to answer questions based on provided documents.

## Core Functionality

The RAG system performs the following operations:

1. **Document Processing**: Loads and processes text documents from a specified directory
2. **Text Chunking**: Splits large documents into smaller, manageable chunks for better retrieval
3. **Embedding Generation**: Creates vector representations of document chunks using sentence transformers
4. **Vector Storage**: Stores document embeddings in a FAISS vector database for efficient similarity search
5. **Question Processing**: Takes user questions and converts them to embeddings
6. **Document Retrieval**: Finds the most relevant document chunks based on semantic similarity
7. **Response Generation**: Uses a local LLM to generate answers based on retrieved context

## Code Structure

```
rag_system/
├── main.py                 # Main entry point for CLI interface
├── llm.py                  # LLM model handling and response generation
├── unit_test.py            # Unit tests for all system components
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── config/                # Configuration settings
│   └── settings.py        # System parameters and paths
├── cache/                 # Cache management
│   └── manager.py         # Caching implementation
├── data/                  # Data storage
│   ├── documents/         # User documents for RAG processing
│   └── cache/             # Cached responses
├── embeddings/            # Embedding and vector storage
│   ├── model.py           # Sentence transformer embedding model
│   ├── storage.py         # FAISS vector database implementation
│   ├── faiss_index.bin    # Persistent FAISS index
│   └── metadata.json      # Document metadata
├── rag/                   # Core RAG engine components
│   ├── engine.py          # Main RAG engine orchestrating components
│   ├── processor.py       # Document loading and chunking
│   └── retriever.py       # Document retrieval and search
```

## Component Descriptions

### 1. Main Components

**main.py**
- Entry point for the CLI interface
- Initializes the RAG engine
- Processes documents automatically on startup
- Provides a question-answering loop for user interaction

**llm.py**
- Handles loading and interaction with the local LLM (Llama 3.2 1B GGUF)
- Formats prompts for the LLM with context and questions
- Generates responses using the LLM's chat completion functionality

### 2. Configuration

**config/settings.py**
- Defines system paths and directories
- Sets model configurations and parameters
- Configures chunking parameters (size, overlap)
- Defines cache settings (TTL, max size)
- Sets LLM generation parameters (temperature, max tokens)

### 3. Core RAG Engine

**rag/engine.py**
- Main orchestrator that connects all RAG components
- Manages document processing workflow
- Handles the complete question-answering pipeline
- Implements caching for improved performance

**rag/processor.py**
- Loads documents from the documents directory
- Implements text chunking with overlap for better context retention
- Splits documents into sentences and forms coherent chunks

**rag/retriever.py**
- Integrates embedding model and FAISS storage
- Performs semantic search for relevant documents
- Manages document indexing and retrieval
- Handles chunking for documents during indexing

### 4. Embedding and Storage

**embeddings/model.py**
- Loads and manages the sentence transformer embedding model
- Generates embeddings for text passages
- Provides embedding dimension information

**embeddings/storage.py**
- Implements FAISS vector database for efficient similarity search
- Manages persistent storage of embeddings and metadata
- Handles vector indexing and search operations

### 5. Caching

**cache/manager.py**
- Implements dual-layer caching (in-memory and disk-based)
- Manages cache expiration and size limits
- Speeds up repeated queries with same questions

## System Workflow

1. **Initialization**:
   - Load configuration settings
   - Initialize embedding model
   - Load or create FAISS index
   - Load LLM model

2. **Document Processing**:
   - Load documents from `data/documents/`
   - Chunk documents into smaller pieces
   - Generate embeddings for each chunk
   - Store embeddings and metadata in FAISS

3. **Question Answering**:
   - Receive user question
   - Check cache for existing answer
   - Generate embedding for question
   - Search FAISS for relevant document chunks
   - Format context with retrieved information
   - Generate prompt for LLM
   - Get response from LLM
   - Cache the result
   - Return answer to user

## Key Features

- **Local Execution**: Runs entirely on local hardware without cloud dependencies
- **CPU Optimization**: Optimized for CPU-only execution
- **Persistent Storage**: FAISS index and metadata are saved between sessions
- **Caching**: Implements caching to improve response times for repeated queries
- **Semantic Search**: Uses embeddings for intelligent document retrieval
- **Modular Design**: Well-organized code structure for maintainability

## Usage

To use the system:
1. Place text documents in the `data/documents/` directory
2. Run `python main.py` to start the CLI interface
3. Ask questions when prompted
4. The system will automatically retrieve relevant information and generate answers