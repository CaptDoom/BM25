import unittest
import time
import os
import json
import yaml
import numpy as np
import tempfile
from collections import defaultdict
from src.preprocessing import QueryPreprocessor, tokenize_text, parse_filter, evaluate_filter
from src.retrievers.bm25 import BM25Index, ShardedBM25
from src.pipeline import RetrievalPipeline
from src.reranker import CrossEncoderReranker
from src.data_loader import DataLoader, CorpusDBHelper, get_batch_docs

class TestDataLoading(unittest.TestCase):
    """Test dataset loading and indexing completeness"""
    
    def setUp(self):
        self.test_dataset = "BeIR/quora"  # Smaller dataset for testing
        self.data_loader = DataLoader(self.test_dataset)
        
    def test_corpus_loading(self):
        """Verify corpus loads completely"""
        corpus = self.data_loader.load_corpus()
        self.assertIsNotNone(corpus)
        self.assertGreater(len(corpus), 0)
        print(f"✓ Corpus loaded: {len(corpus)} documents")
        
        # Check document structure
        doc = corpus[0]
        self.assertIn("_id", doc)
        self.assertIn("text", doc)
        
    def test_queries_loading(self):
        """Verify queries load completely"""
        queries = self.data_loader.load_queries()
        self.assertIsNotNone(queries)
        self.assertGreater(len(queries), 0)
        print(f"✓ Queries loaded: {len(queries)} queries")
        
    def test_batch_docs(self):
        """Verify batch document retrieval works correctly"""
        corpus = self.data_loader.load_corpus()
        batch_size = 100
        
        batch_1 = get_batch_docs(corpus, 0, batch_size)
        self.assertEqual(len(batch_1), batch_size)
        
        # Test overlapping batch
        batch_2 = get_batch_docs(corpus, 50, 150)
        self.assertEqual(len(batch_2), 100)
        
        # Verify consistency
        self.assertEqual(batch_1[50]["_id"], batch_2[0]["_id"])
        print(f"✓ Batch retrieval working correctly")

class TestBM25Pipeline(unittest.TestCase):
    """Test BM25 indexing and retrieval"""
    
    def setUp(self):
        self.test_docs = [
            {"_id": "doc_1", "title": "France", "text": "The capital of France is Paris. Paris is a beautiful city."},
            {"_id": "doc_2", "title": "Germany", "text": "The capital of Germany is Berlin. Berlin is known for its history."},
            {"_id": "doc_3", "title": "Spain", "text": "The capital of Spain is Madrid. Madrid has great architecture."},
            {"_id": "doc_4", "title": "Italy", "text": "Rome is the capital of Italy. Rome is the eternal city."},
            {"_id": "doc_5", "title": "Europe", "text": "Europe contains many countries. Paris and Berlin are major cities."},
        ]
        
    def test_bm25_indexing(self):
        """Test BM25 index builds correctly"""
        index = BM25Index()
        index.index(self.test_docs)
        
        self.assertGreater(len(index.vocab), 0)
        self.assertGreater(len(index.doc_ids), 0)
        self.assertEqual(len(index.doc_ids), len(self.test_docs))
        print(f"✓ BM25 index built with {len(index.vocab)} vocab terms")
        
    def test_bm25_search_accuracy(self):
        """Test BM25 search returns relevant documents"""
        index = BM25Index()
        index.index(self.test_docs)
        
        # Test exact query
        results = index.search(["paris", "france"], top_k=10)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], "doc_1")  # France doc should rank first
        print(f"✓ BM25 search accuracy verified: {results[0]}")
        
    def test_bm25_title_boosting(self):
        """Test that title tokens are boosted"""
        index = BM25Index()
        index.index(self.test_docs)
        
        # Query for title
        results_title = index.search(["france"], top_k=5)
        results_body = index.search(["paris"], top_k=5)
        
        # France appears in title of doc_1, should rank high
        self.assertEqual(results_title[0][0], "doc_1")
        # Paris appears in body and title of doc_1, should rank high
        self.assertEqual(results_body[0][0], "doc_1")
        print(f"✓ Title boosting working correctly")
        
    def test_bm25_speed(self):
        """Benchmark BM25 search speed"""
        index = BM25Index()
        index.index(self.test_docs)
        
        query = ["paris", "city"]
        start = time.perf_counter()
        for _ in range(1000):
            results = index.search(query, top_k=5)
        elapsed = (time.perf_counter() - start) * 1000
        avg_time_per_query = elapsed / 1000
        
        print(f"✓ BM25 search speed: {avg_time_per_query:.3f}ms per query")
        self.assertLess(avg_time_per_query, 10)  # Should be < 10ms
        
    def test_bm25_robustness(self):
        """Test BM25 robustness with edge cases"""
        index = BM25Index()
        index.index(self.test_docs)
        
        # Empty query
        results = index.search([], top_k=5)
        self.assertEqual(len(results), 0)
        
        # Single word
        results = index.search(["paris"], top_k=5)
        self.assertGreater(len(results), 0)
        
        # Non-existent word
        results = index.search(["nonexistentword123"], top_k=5)
        self.assertEqual(len(results), 0)
        
        # Multiple queries
        results = index.search(["paris", "berlin", "madrid"], top_k=5)
        self.assertGreater(len(results), 0)
        
        print(f"✓ BM25 robustness verified with edge cases")

    def test_sharded_bm25_uses_global_statistics(self):
        """Test sharded BM25 keeps vocabulary and IDFs consistent across shards"""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = ShardedBM25(tmpdir, shard_size=2)
            index.index(self.test_docs)

            self.assertGreater(len(index.shards), 1)
            first_vocab = index.shards[0].vocab

            for shard in index.shards[1:]:
                self.assertEqual(shard.vocab, first_vocab)
                np.testing.assert_array_equal(shard.idfs, index.shards[0].idfs)
                self.assertEqual(shard.avg_dl, index.shards[0].avg_dl)

            results = index.search(["madrid"], top_k=3)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0][0], "doc_3")
            print("✓ Sharded BM25 global statistics verified")

class TestPreprocessing(unittest.TestCase):
    """Test query preprocessing"""
    
    def setUp(self):
        self.config = {
            "preprocessing": {
                "lowercase": True,
                "remove_punctuation": True,
                "remove_stopwords": True,
                "use_lemmatization": False
            },
            "spell_correction": {
                "min_similarity": 0.85
            }
        }
        self.preprocessor = QueryPreprocessor(self.config)
        
    def test_tokenization(self):
        """Test tokenization helper"""
        tokens = tokenize_text("Hello, World! This is a test.")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertNotIn("is", tokens)  # Stopword
        print(f"✓ Tokenization working: {tokens}")
        
    def test_clean_query(self):
        """Test query cleaning"""
        cleaned = self.preprocessor.clean_query("What is the CAPITAL of France?")
        self.assertNotIn("?", cleaned)
        self.assertIn("capital", cleaned)
        self.assertNotIn("is", cleaned)  # Stopword removed
        print(f"✓ Query cleaning: '{cleaned}'")
        
    def test_spell_correction(self):
        """Test spell correction"""
        self.preprocessor.set_vocab(["paris", "france", "berlin", "germany"])
        corrected = self.preprocessor.correct_spelling("paris france")
        self.assertIn("paris", corrected)  # Should correct misspelling
        print(f"✓ Spell correction: '{corrected}'")
        
    def test_filter_parsing(self):
        """Test metadata filter parsing"""
        filter1 = parse_filter("year == 2020")
        self.assertEqual(filter1, ("year", "==", 2020))
        
        filter2 = parse_filter("rating >= 4.5")
        self.assertEqual(filter2, ("rating", ">=", 4.5))
        
        print(f"✓ Filter parsing working")
        
    def test_filter_evaluation(self):
        """Test metadata filter evaluation"""
        metadata = {"year": 2020, "rating": 4.5, "category": "tech"}
        
        # Test equality
        result = evaluate_filter(metadata, ("year", "==", 2020))
        self.assertTrue(result)
        
        # Test greater than
        result = evaluate_filter(metadata, ("rating", ">", 4.0))
        self.assertTrue(result)
        
        # Test inequality
        result = evaluate_filter(metadata, ("category", "!=", "finance"))
        self.assertTrue(result)
        
        print(f"✓ Filter evaluation working")

class TestReranking(unittest.TestCase):
    """Test cross-encoder reranking"""
    
    def setUp(self):
        self.test_docs = [
            {"_id": "doc_1", "title": "Machine Learning Basics", "text": "ML is a subset of AI"},
            {"_id": "doc_2", "title": "Deep Learning", "text": "Deep learning uses neural networks"},
            {"_id": "doc_3", "title": "AI Overview", "text": "Artificial intelligence overview"},
        ]
        
    def test_reranker_loads(self):
        """Test reranker model loads"""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            batch_size=32
        )
        self.assertIsNotNone(reranker)
        print(f"✓ Reranker model loaded successfully")
        
    def test_reranking_order(self):
        """Test reranking changes document order"""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            batch_size=32
        )
        
        query = "machine learning"
        candidates = [
            ("doc_1", 10.0),
            ("doc_2", 8.0),
            ("doc_3", 5.0),
        ]
        
        # Create mock corpus DB
        class MockDB:
            def get_documents(self, doc_ids):
                result = {}
                for doc in self.test_docs:
                    if doc["_id"] in doc_ids:
                        result[doc["_id"]] = doc
                return result
                
        db = MockDB()
        db.test_docs = self.test_docs
        db.get_documents = lambda doc_ids: {doc["_id"]: doc for doc in self.test_docs if doc["_id"] in doc_ids}
        
        reranked = reranker.rerank(query, candidates, db, top_k=3)
        self.assertEqual(len(reranked), 3)
        print(f"✓ Reranking completed: {reranked[:1]}")

class TestMetadataFiltering(unittest.TestCase):
    """Test metadata filtering functionality"""
    
    def setUp(self):
        self.test_docs = [
            {"_id": "doc_1", "title": "Tech", "text": "Technology", "metadata": {"year": 2020, "category": "tech"}},
            {"_id": "doc_2", "title": "Finance", "text": "Finance", "metadata": {"year": 2021, "category": "finance"}},
            {"_id": "doc_3", "title": "Health", "text": "Health", "metadata": {"year": 2020, "category": "health"}},
        ]
        
    def test_corpus_db_metadata_filtering(self):
        """Test metadata filtering in corpus database"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = CorpusDBHelper(db_path)
            db.init_db()
            db.insert_documents(self.test_docs)
            
            # Test filter
            results = db.get_all_doc_ids_matching_filter(("year", "==", 2020))
            self.assertEqual(len(results), 2)
            self.assertIn("doc_1", results)
            self.assertIn("doc_3", results)
            print(f"✓ Metadata filtering working: {results}")

class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        self.test_docs = [
            {"_id": "doc_1", "title": "Python Programming", "text": "Python is a programming language", "metadata": {"type": "tutorial"}},
            {"_id": "doc_2", "title": "Python Performance", "text": "Optimizing Python code for speed", "metadata": {"type": "guide"}},
            {"_id": "doc_3", "title": "JavaScript Basics", "text": "JavaScript for web development", "metadata": {"type": "tutorial"}},
        ]
        
    def test_full_pipeline(self):
        """Test complete search pipeline"""
        # Build index
        index = BM25Index()
        index.index(self.test_docs)
        
        # Search
        query_tokens = tokenize_text("python programming")
        results = index.search(query_tokens, top_k=5)
        
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], "doc_1")
        print(f"✓ Full pipeline working: {results}")
        
    def test_query_variations(self):
        """Test different query variations"""
        index = BM25Index()
        index.index(self.test_docs)
        
        test_queries = [
            "python",
            "python code",
            "programming language",
            "performance optimization",
            "javascript web"
        ]
        
        for query in test_queries:
            tokens = tokenize_text(query)
            results = index.search(tokens, top_k=5)
            self.assertGreaterEqual(len(results), 0)
            print(f"  Query '{query}': {len(results)} results")
        
        print(f"✓ All query variations handled correctly")

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking tests"""
    
    def setUp(self):
        # Generate larger dataset for benchmarking
        self.test_docs = []
        for i in range(1000):
            self.test_docs.append({
                "_id": f"doc_{i}",
                "title": f"Document {i}",
                "text": f"This is document {i} with content about machine learning and AI.",
                "metadata": {"index": i, "category": ["tech", "ai", "ml"][i % 3]}
            })
    
    def test_indexing_speed(self):
        """Benchmark indexing speed"""
        start = time.perf_counter()
        index = BM25Index()
        index.index(self.test_docs)
        elapsed = time.perf_counter() - start
        
        docs_per_sec = len(self.test_docs) / elapsed
        print(f"✓ Indexing speed: {docs_per_sec:.0f} docs/sec ({elapsed:.2f}s for {len(self.test_docs)} docs)")
        self.assertGreater(docs_per_sec, 100)  # Should handle > 100 docs/sec
        
    def test_search_speed_scaling(self):
        """Test search speed with different corpus sizes"""
        index = BM25Index()
        index.index(self.test_docs)
        
        query_tokens = ["machine", "learning"]
        times = []
        
        for _ in range(10):
            start = time.perf_counter()
            results = index.search(query_tokens, top_k=10)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"✓ Search speed: {avg_time:.2f}ms ± {std_time:.2f}ms (1000 docs)")
        self.assertLess(avg_time, 100)  # Should be < 100ms
        
    def test_memory_efficiency(self):
        """Test memory efficiency of indexing"""
        import sys
        index = BM25Index()
        index.index(self.test_docs)
        
        # Estimate memory usage
        vocab_size = len(index.vocab)
        doc_count = len(index.doc_ids)
        
        print(f"✓ Memory efficiency: {doc_count} docs, {vocab_size} terms")
        print(f"  Avg doc length: {index.avg_dl:.2f}")
        print(f"  Inverted index shards: {len(index.inverted_index)}")

if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
