import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.engine import RAGEngine

def main():
    print("Initializing Optimized Local RAG System...")
    
    # Create RAG engine
    rag_engine = RAGEngine()
    
    # Process documents
    rag_engine.process_documents()
    
    # QA loop
    print("\nOptimized Local RAG System")
    print("Type 'exit' to quit")
    
    while True:
        user_input = input("\nQuestion: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
            
        # Get response from RAG engine
        answer = rag_engine.query(user_input)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()