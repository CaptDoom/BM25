import unittest
import os
import shutil
import tempfile
import numpy as np
from src.retrievers.bm25 import BM25Index, build_metadata_and_bitmaps, CppBM25Engine
from src.data_loader import CorpusDBHelper

class TestMetadataFiltering(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the index
        self.test_dir = tempfile.mkdtemp()
        
        # Define mock documents representing different datasets, lengths, title presences, code presences
        self.mock_docs = [
            {
                "_id": "doc_1", 
                "title": "Introduction to Python", 
                "text": "Python is an easy to learn programming language used for web dev and AI.", 
                "metadata": {"dataset": "quora"}
            },
            {
                "_id": "doc_2", 
                "title": "", # No title
                "text": "This text is very short.", 
                "metadata": {"dataset": "quora"}
            },
            {
                "_id": "doc_3", 
                "title": "C++ std::vector example", 
                "text": "Let's write some code: `std::vector<int> v;`. It is extremely fast and high performance.", 
                "metadata": {"dataset": "cqadupstack"}
            },
            {
                "_id": "doc_4", 
                "title": "Scientific paper on retrieval", 
                "text": "In this paper we present a novel BM25 search engine with Roaring bitmaps. " * 30, # Long doc
                "metadata": {"dataset": "scidocs"}
            },
            {
                "_id": "doc_5", 
                "title": "FEVER Claim Verification", 
                "text": "This is a fact verification claim doc.", 
                "metadata": {"dataset": "fever"}
            }
        ]
        
        # 1. Initialize SQLite Database
        db_path = os.path.join(self.test_dir, "corpus.db")
        self.db_helper = CorpusDBHelper(db_path)
        self.db_helper.init_db()
        self.db_helper.insert_documents(self.mock_docs)
        
        # 2. Build BM25 index
        self.index = BM25Index()
        self.index.index(self.mock_docs)
        self.index.save(self.test_dir)
        
        # Load the index back (which automatically sets up the C++ engine and/or bitmaps)
        self.index.load(self.test_dir)

    def tearDown(self):
        # Clean up C++ engine before deleting files
        if self.index.cpp_engine:
            self.index.cpp_engine = None
        shutil.rmtree(self.test_dir)

    def test_metadata_generation(self):
        """Verify metadata binary, bitmaps binary, and text files exist"""
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "doc_metadata.bin")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "bitmaps.bin")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "bitmaps.txt")))

    def test_cpp_engine_loads(self):
        """Verify C++ engine compiles and loads successfully"""
        self.assertIsNotNone(self.index.cpp_engine)
        self.assertIsNotNone(self.index.cpp_engine.engine_ptr)

    def test_cpp_search_without_filters(self):
        """Verify C++ search returns exact same scores as Python when no filters are present"""
        query_tokens = ["python", "programming"]
        
        # Python search (bypass C++ by setting doc_mask with all document IDs)
        py_results = self.index.search(query_tokens, top_k=5, doc_mask=set(self.index.doc_ids))
        
        # C++ search
        cpp_results = self.index.search(query_tokens, top_k=5)
        
        self.assertTrue(len(py_results) > 0)
        self.assertEqual(len(cpp_results), len(py_results))
        for cpp_r, py_r in zip(cpp_results, py_results):
            self.assertEqual(cpp_r[0], py_r[0]) # Same Doc ID
            self.assertAlmostEqual(cpp_r[1], py_r[1], places=4) # Same Score

    def test_cpp_search_with_dataset_filter(self):
        """Verify C++ search honors dataset filter"""
        # doc_1 is 'quora' (ds_1), doc_3 is 'cqadupstack' (ds_4)
        query_tokens = ["code", "programming", "python", "vector"]
        
        # Filter: quora (should return doc_1 only, not doc_3 which is cqadupstack)
        results = self.index.search(query_tokens, top_k=5, filter_names=["ds_1"])
        doc_ids = [r[0] for r in results]
        self.assertIn("doc_1", doc_ids)
        self.assertNotIn("doc_3", doc_ids)
        
        # Filter: cqadupstack (should return doc_3 only)
        results = self.index.search(query_tokens, top_k=5, filter_names=["ds_4"])
        doc_ids = [r[0] for r in results]
        self.assertIn("doc_3", doc_ids)
        self.assertNotIn("doc_1", doc_ids)

    def test_cpp_search_with_length_filter(self):
        """Verify C++ search honors length filter"""
        # doc_4 is extremely long (>2000 chars)
        query_tokens = ["retrieval", "engine"]
        
        results = self.index.search(query_tokens, top_k=5, filter_names=["len_long"])
        doc_ids = [r[0] for r in results]
        self.assertEqual(len(doc_ids), 1)
        self.assertEqual(doc_ids[0], "doc_4")

    def test_cpp_search_with_title_filter(self):
        """Verify C++ search honors title filter"""
        # doc_2 has no title (has_title_0)
        query_tokens = ["text", "short"]
        
        results = self.index.search(query_tokens, top_k=5, filter_names=["has_title_0"])
        doc_ids = [r[0] for r in results]
        self.assertIn("doc_2", doc_ids)
        
        # Check has_title_1 filter doesn't return doc_2
        results = self.index.search(query_tokens, top_k=5, filter_names=["has_title_1"])
        doc_ids = [r[0] for r in results]
        self.assertNotIn("doc_2", doc_ids)

    def test_cpp_search_with_code_filter(self):
        """Verify C++ search honors code presence filter"""
        # doc_3 has code snippet
        query_tokens = ["code", "vector"]
        results = self.index.search(query_tokens, top_k=5, filter_names=["has_code_1"])
        doc_ids = [r[0] for r in results]
        self.assertIn("doc_3", doc_ids)

    def test_python_fallback(self):
        """Verify Python-only fallback works correctly when C++ engine is disabled"""
        # Temporarily disable C++ engine
        original_cpp_engine = self.index.cpp_engine
        self.index.cpp_engine = None
        
        try:
            query_tokens = ["python", "programming"]
            
            # Search with filter using Python fallback
            results = self.index.search(query_tokens, top_k=5, filter_names=["ds_1"])
            doc_ids = [r[0] for r in results]
            self.assertIn("doc_1", doc_ids)
            self.assertNotIn("doc_3", doc_ids)
        finally:
            self.index.cpp_engine = original_cpp_engine

if __name__ == "__main__":
    unittest.main()
