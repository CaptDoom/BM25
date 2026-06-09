import unittest
from src.preprocessing import QueryPreprocessor
from src.retrievers.bm25 import BM25Index
from src.retrievers.hybrid import RRFHybrid

class TestPipeline(unittest.TestCase):
    def test_clean_query(self):
        config = {
            "preprocessing": {
                "lowercase": True,
                "remove_punctuation": True,
                "remove_stopwords": True,
                "use_lemmatization": False
            }
        }
        preprocessor = QueryPreprocessor(config)
        cleaned = preprocessor.clean_query("What is the capital of France?")
        self.assertEqual(cleaned, "capital france")

    def test_spell_correction(self):
        config = {
            "spell_correction": {
                "min_similarity": 0.85
            }
        }
        preprocessor = QueryPreprocessor(config)
        preprocessor.set_vocab(["france", "capital", "germany", "berlin"])
        corrected = preprocessor.correct_spelling("frnce capital")
        self.assertEqual(corrected, "france capital")

    def test_bm25_index(self):
        documents = [
            {"_id": "doc_1", "title": "France", "text": "The capital of France is Paris."},
            {"_id": "doc_2", "title": "Germany", "text": "The capital of Germany is Berlin."}
        ]
        index = BM25Index()
        index.index(documents)
        
        results = index.search(["france"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc_1")

    def test_rrf_fusion(self):
        bm25_res = [("doc_1", 10.0), ("doc_2", 8.0)]
        dense_res = [("doc_2", 0.1), ("doc_1", 0.5)]
        
        fuser = RRFHybrid(k=60)
        fused = fuser.fuse(bm25_res, dense_res, top_k=2)
        
        fused_ids = [d[0] for d in fused]
        self.assertIn("doc_1", fused_ids)
        self.assertIn("doc_2", fused_ids)

if __name__ == "__main__":
    unittest.main()
