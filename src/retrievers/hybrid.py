class RRFHybrid:
    def __init__(self, k=60, weights=None):
        self.k = k
        self.weights = weights if weights else {"bm25": 1.0, "dense": 1.0}
        
    def fuse(self, bm25_results, dense_results, top_k=100):
        rrf_scores = {}
        
        # Process BM25 results
        for rank, (doc_id, _) in enumerate(bm25_results):
            weight = self.weights.get("bm25", 1.0)
            score = weight / (self.k + (rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
            
        # Process Dense results
        for rank, (doc_id, _) in enumerate(dense_results):
            weight = self.weights.get("dense", 1.0)
            score = weight / (self.k + (rank + 1))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
            
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
