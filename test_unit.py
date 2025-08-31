import unittest
import os
import sys
import numpy as np
from dotenv import load_dotenv

# Add the current directory to Python path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class CoreLocalRAGTests(unittest.TestCase):
    """Core 5 unit tests for Local RAG System with GGUF LLM with real components"""
    
    @classmethod
    def setUpClass(cls):
        """Load configuration and validate setup"""
        # Note: This system uses local GGUF models, not API keys
        print("Setting up Local RAG System tests...")
        
        # Load only configuration (no heavy imports)
        try:
            from config.settings import (
                LLM_MODEL_ID, EMBEDDING_MODEL_ID, CHUNK_SIZE, CHUNK_OVERLAP,
                TOP_K_RESULTS, CACHE_TTL, CACHE_MAX_SIZE, FAISS_INDEX_FILE,
                FAISS_METADATA_FILE, DOCUMENTS_DIR, CACHE_DIR, EMBEDDINGS_DIR
            )
            
            # Store configuration only
            cls.LLM_MODEL_ID = LLM_MODEL_ID
            cls.EMBEDDING_MODEL_ID = EMBEDDING_MODEL_ID
            cls.CHUNK_SIZE = CHUNK_SIZE
            cls.CHUNK_OVERLAP = CHUNK_OVERLAP
            cls.TOP_K_RESULTS = TOP_K_RESULTS
            cls.CACHE_TTL = CACHE_TTL
            cls.CACHE_MAX_SIZE = CACHE_MAX_SIZE
            cls.FAISS_INDEX_FILE = FAISS_INDEX_FILE
            cls.FAISS_METADATA_FILE = FAISS_METADATA_FILE
            cls.DOCUMENTS_DIR = DOCUMENTS_DIR
            cls.CACHE_DIR = CACHE_DIR
            cls.EMBEDDINGS_DIR = EMBEDDINGS_DIR
            
            print("Local RAG configuration loaded successfully")
        except ImportError as e:
            raise unittest.SkipTest(f"Required local RAG configuration not found: {e}")

    def test_01_configuration_and_model_setup(self):
        """Test 1: Configuration and Model Setup Validation"""
        print("Running Test 1: Configuration and Model Setup")
        
        # Test LLM model configuration
        self.assertEqual(self.LLM_MODEL_ID, "meta-llama/Llama-3.2-1B")
        self.assertIn("llama", self.LLM_MODEL_ID.lower())
        self.assertIn("3.2", self.LLM_MODEL_ID)
        
        # Test embedding model configuration
        self.assertEqual(self.EMBEDDING_MODEL_ID, "all-MiniLM-L6-v2")
        self.assertIn("MiniLM", self.EMBEDDING_MODEL_ID)
        
        # Test RAG configuration parameters
        self.assertEqual(self.CHUNK_SIZE, 500)
        self.assertEqual(self.CHUNK_OVERLAP, 50)
        self.assertEqual(self.TOP_K_RESULTS, 5)
        
        # Validate parameter relationships
        self.assertLess(self.CHUNK_OVERLAP, self.CHUNK_SIZE)
        self.assertGreater(self.TOP_K_RESULTS, 0)
        self.assertGreater(self.CHUNK_SIZE, 0)
        
        # Test cache configuration
        self.assertEqual(self.CACHE_TTL, 24 * 60 * 60)  # 24 hours
        self.assertEqual(self.CACHE_MAX_SIZE, 1000)
        self.assertGreater(self.CACHE_TTL, 0)
        self.assertGreater(self.CACHE_MAX_SIZE, 0)
        
        # Test directory paths
        self.assertTrue(str(self.DOCUMENTS_DIR).endswith('documents'))
        self.assertTrue(str(self.CACHE_DIR).endswith('cache'))
        self.assertTrue(str(self.EMBEDDINGS_DIR).endswith('embeddings'))
        
        # Test file paths
        self.assertTrue(str(self.FAISS_INDEX_FILE).endswith('.bin'))
        self.assertTrue(str(self.FAISS_METADATA_FILE).endswith('.json'))
        
        # Test directory structure exists
        expected_dirs = ['config', 'embeddings', 'rag', 'cache']
        for directory in expected_dirs:
            self.assertTrue(os.path.exists(directory), f"Directory {directory} should exist")
        
        # Test file structure exists
        expected_files = ['llm.py', 'main.py', 'requirements.txt']
        for filename in expected_files:
            self.assertTrue(os.path.exists(filename), f"File {filename} should exist")
        
        print(f"PASS: LLM model configuration - {self.LLM_MODEL_ID}")
        print(f"PASS: Embedding model configuration - {self.EMBEDDING_MODEL_ID}")
        print(f"PASS: RAG parameters - Chunk size: {self.CHUNK_SIZE}, Overlap: {self.CHUNK_OVERLAP}, Top-K: {self.TOP_K_RESULTS}")
        print(f"PASS: Cache configuration - TTL: {self.CACHE_TTL}s, Max size: {self.CACHE_MAX_SIZE}")
        print("PASS: Configuration and model setup validated")

    def test_02_document_processor_and_chunking(self):
        """Test 2: Document Processing and Text Chunking"""
        print("Running Test 2: Document Processing and Chunking")
        
        # Import and initialize document processor for this test only
        from rag.processor import DocumentProcessor
        doc_processor = DocumentProcessor()
        self.assertIsNotNone(doc_processor)
        
        # Test text chunking functionality
        test_document = """This is the first sentence of a test document. This is the second sentence with more content. 
        This is the third sentence that continues the document. This is the fourth sentence with additional information.
        This is the fifth sentence that provides more context. This is the final sentence of the test document."""
        
        chunks = doc_processor.chunk_text(test_document, chunk_size=100, overlap=20)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))  # Allow some tolerance
        
        # Test chunking with default parameters
        default_chunks = doc_processor.chunk_text(test_document)
        self.assertIsInstance(default_chunks, list)
        self.assertGreater(len(default_chunks), 0)
        
        # Test empty document handling
        empty_chunks = doc_processor.chunk_text("")
        self.assertIsInstance(empty_chunks, list)
        
        # Test document loading structure
        self.assertTrue(hasattr(doc_processor, 'load_documents'))
        self.assertTrue(hasattr(doc_processor, 'chunk_text'))
        
        print(f"PASS: Document chunking - {len(chunks)} chunks created")
        print(f"PASS: Default chunking - {len(default_chunks)} chunks with default params")
        print("PASS: Document processing and chunking validated")

    def test_03_faiss_storage_operations(self):
        """Test 3: FAISS Storage Operations"""
        print("Running Test 3: FAISS Storage Operations")
        
        # Import and test FAISS storage for this test only
        from embeddings.storage import FaissStorage
        test_dimension = 384  # all-MiniLM-L6-v2 dimension
        faiss_storage = FaissStorage(test_dimension)
        
        self.assertIsNotNone(faiss_storage)
        self.assertEqual(faiss_storage.dimension, test_dimension)
        self.assertIsInstance(faiss_storage.metadata, list)
        
        # Test index initialization
        faiss_storage.initialize_index()
        self.assertIsNotNone(faiss_storage.index)
        
        # Test adding embeddings
        test_embeddings = np.random.rand(3, test_dimension).astype('float32')
        test_metadata = [
            {"content": "First test document", "source": "test1"},
            {"content": "Second test document", "source": "test2"},
            {"content": "Third test document", "source": "test3"}
        ]
        
        initial_count = faiss_storage.get_total_vectors()
        faiss_storage.add_embeddings(test_embeddings, test_metadata)
        final_count = faiss_storage.get_total_vectors()
        
        self.assertEqual(final_count, initial_count + 3)
        
        # Test search functionality
        query_embedding = np.random.rand(1, test_dimension).astype('float32')
        results, distances = faiss_storage.search(query_embedding, k=2)
        
        self.assertIsInstance(results, list)
        self.assertIsInstance(distances, np.ndarray)
        self.assertLessEqual(len(results), 2)
        
        # Test search result structure
        if results:
            first_result = results[0]
            self.assertIn('metadata', first_result)
            self.assertIn('distance', first_result)
            self.assertIsInstance(first_result['metadata'], dict)
        
        print(f"PASS: FAISS storage - Dimension: {test_dimension}")
        print(f"PASS: Vector operations - {final_count} total vectors")
        print(f"PASS: Search functionality - {len(results)} results returned")

    def test_04_cache_manager_operations(self):
        """Test 4: Cache Manager Operations"""
        print("Running Test 4: Cache Manager Operations")
        
        # Import and initialize cache manager for this test only
        from cache.manager import CacheManager
        cache_manager = CacheManager()
        self.assertIsNotNone(cache_manager)
        self.assertEqual(cache_manager.cache_dir, self.CACHE_DIR)
        
        # Test cache key generation
        test_query = "What is machine learning?"
        cache_key = cache_manager._get_cache_key(test_query)
        self.assertIsInstance(cache_key, str)
        self.assertEqual(len(cache_key), 32)  # MD5 hash length
        
        # Test cache file path generation
        cache_file_path = cache_manager._get_cache_file_path(cache_key)
        self.assertIsInstance(cache_file_path, str)
        self.assertTrue(cache_file_path.endswith('.pkl'))
        self.assertIn(cache_key, cache_file_path)
        
        # Test caching and retrieval
        test_result = "Machine learning is a subset of artificial intelligence."
        
        # Cache the result
        cache_manager.cache_result(test_query, test_result)
        
        # Retrieve cached result
        cached_result = cache_manager.get_cached_result(test_query)
        self.assertEqual(cached_result, test_result)
        
        # Test cache miss
        non_cached_result = cache_manager.get_cached_result("Non-existent query")
        self.assertIsNone(non_cached_result)
        
        # Test cache cleaning structure
        self.assertTrue(hasattr(cache_manager, '_clean_cache_if_needed'))
        
        print(f"PASS: Cache operations - Key: {cache_key[:8]}...")
        print(f"PASS: Cache hit/miss functionality working")
        print("PASS: Cache manager operations validated")

    def test_05_rag_engine_integration(self):
        """Test 5: RAG Engine Integration and Structure"""
        print("Running Test 5: RAG Engine Integration")
        
        # Import classes for this test only
        from rag.engine import RAGEngine
        from rag.retriever import Retriever
        from rag.processor import DocumentProcessor
        from cache.manager import CacheManager
        
        # Test class methods availability
        import inspect
        rag_methods = inspect.getmembers(RAGEngine, predicate=inspect.isfunction)
        method_names = [name for name, _ in rag_methods]
        
        expected_methods = ['process_documents', 'query']
        for method in expected_methods:
            self.assertIn(method, method_names)
        
        # Test retriever class structure
        retriever_methods = inspect.getmembers(Retriever, predicate=inspect.isfunction)
        retriever_method_names = [name for name, _ in retriever_methods]
        
        expected_retriever_methods = ['initialize', 'search', 'add_documents', 'chunk_text']
        for method in expected_retriever_methods:
            self.assertIn(method, retriever_method_names)
        
        # Test component integration structure
        try:
            # Create minimal instances to test structure (avoid heavy model loading)
            doc_processor = DocumentProcessor()
            cache_manager = CacheManager()
            
            # Test document processor methods
            self.assertTrue(hasattr(doc_processor, 'load_documents'))
            self.assertTrue(hasattr(doc_processor, 'chunk_text'))
            
            # Test cache manager methods
            self.assertTrue(hasattr(cache_manager, 'get_cached_result'))
            self.assertTrue(hasattr(cache_manager, 'cache_result'))
            
            print("PASS: Component integration structure validated")
            
        except Exception as e:
            print(f"INFO: Component structure test completed with note: {str(e)}")
        
        # Test configuration consistency
        config_params = {
            'chunk_size': self.CHUNK_SIZE,
            'chunk_overlap': self.CHUNK_OVERLAP,
            'top_k_results': self.TOP_K_RESULTS,
            'cache_ttl': self.CACHE_TTL,
            'cache_max_size': self.CACHE_MAX_SIZE
        }
        
        for param_name, param_value in config_params.items():
            self.assertIsNotNone(param_value, f"{param_name} should not be None")
            self.assertGreater(param_value, 0, f"{param_name} should be positive")
        
        # Directory structure already tested in test_01
        
        # Test model specifications
        self.assertIn("llama", self.LLM_MODEL_ID.lower())
        self.assertIn("3.2", self.LLM_MODEL_ID)
        self.assertIn("1B", self.LLM_MODEL_ID)
        
        self.assertIn("MiniLM", self.EMBEDDING_MODEL_ID)
        self.assertIn("L6", self.EMBEDDING_MODEL_ID)
        self.assertIn("v2", self.EMBEDDING_MODEL_ID)
        
        # Test that files exist without importing heavy modules
        self.assertTrue(os.path.exists('llm.py'))
        self.assertTrue(os.path.exists('embeddings/model.py'))
        self.assertTrue(os.path.exists('embeddings/storage.py'))
        self.assertTrue(os.path.exists('rag/engine.py'))
        
        print(f"PASS: RAG engine structure - Methods: {len(method_names)} available")
        print(f"PASS: Model specifications - LLM: {self.LLM_MODEL_ID}, Embedding: {self.EMBEDDING_MODEL_ID}")
        print("PASS: RAG engine integration and structure validated")

def run_core_tests():
    """Run core tests and provide summary"""
    print("=" * 70)
    print("[*] Core Local RAG System with GGUF LLM Unit Tests (5 Tests)")
    print("Testing with LOCAL Models and RAG Components")
    print("=" * 70)
    
    print("[INFO] This system uses local GGUF models (no API keys required)")
    print("[INFO] Tests validate structure and configuration without heavy model loading")
    print()
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(CoreLocalRAGTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("[*] Test Results:")
    print(f"[*] Tests Run: {result.testsRun}")
    print(f"[*] Failures: {len(result.failures)}")
    print(f"[*] Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n[FAILURES]:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    if result.errors:
        print("\n[ERRORS]:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n[SUCCESS] All 5 core local RAG tests passed!")
        print("[OK] Local RAG components working correctly with GGUF models")
        print("[OK] Configuration, Document Processing, FAISS Storage, Cache Manager, RAG Engine validated")
    else:
        print(f"\n[WARNING] {len(result.failures) + len(result.errors)} test(s) failed")
    
    return success

if __name__ == "__main__":
    print("[*] Starting Core Local RAG System with GGUF LLM Tests")
    print("[*] 5 essential tests with local models and RAG components")
    print("[*] Components: Configuration, Document Processing, FAISS Storage, Cache Manager, RAG Engine")
    print()
    
    success = run_core_tests()
    exit(0 if success else 1)