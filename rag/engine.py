import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.processor import DocumentProcessor
from rag.retriever import Retriever
from cache.manager import CacheManager
from llm import generate_response
import time

class RAGEngine:
    def __init__(self):
        """Initialize RAG engine"""
        self.document_processor = DocumentProcessor()
        self.retriever = Retriever()
        self.cache_manager = CacheManager()
        
        # Initialize retriever
        self.retriever.initialize()
        
    def process_documents(self):
        """Process and index all documents"""
        print("Loading documents...")
        documents = self.document_processor.load_documents()
        
        if not documents:
            print("No documents found to process")
            return
            
        print(f"Processing {len(documents)} documents...")
        self.retriever.add_documents(documents)
        print("Document processing complete")
        
    def query(self, question):
        """Process a query through the RAG pipeline"""
        # Check cache first
        cached_result = self.cache_manager.get_cached_result(question)
        if cached_result:
            print("Using cached result")
            return cached_result
            
        # Retrieve relevant documents
        print("Retrieving relevant documents...")
        results, distances = self.retriever.search(question)
        
        # Format context
        context = "\n".join([result['metadata']['content'] for result in results])
        
        # Generate prompt with context
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"
            
        # Generate response using LLM
        print("Generating response...")
        start_time = time.time()
        answer = generate_response(prompt)
        end_time = time.time()
        
        print(f"Response generated in {end_time - start_time:.2f} seconds")
        
        # Cache the result
        self.cache_manager.cache_result(question, answer)
        
        return answer