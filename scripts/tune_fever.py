import os
import sys
import yaml
import math
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RetrievalPipeline
from src.data_loader import DataLoader

def main():
    dataset_name = "BeIR/fever"
    index_dir = "src/index/BeIR_fever"
    
    if not os.path.exists(index_dir):
        print(f"Error: Index directory {index_dir} does not exist.")
        return
        
    print("Loading pipeline...")
    pipeline = RetrievalPipeline("config/config.yaml", index_dir)
    pipeline.load_indexes()
    
    print("Loading queries and qrels...")
    loader = DataLoader(dataset_name)
    queries = loader.load_queries()
    qrels = loader.load_qrels()
    
    qrels_dict = {}
    for row in qrels:
        q_id = str(row['query-id'])
        doc_id = str(row['corpus-id'])
        score = int(row['score'])
        if score > 0:
            qrels_dict.setdefault(q_id, {})[doc_id] = score
            
    eval_queries = [
        {"id": q["_id"], "text": q["text"]}
        for q in queries if q["_id"] in qrels_dict
    ]
    
    # We will evaluate on a subset of 100 queries for speed
    limit = min(100, len(eval_queries))
    eval_queries = eval_queries[:limit]
    print(f"Loaded {len(eval_queries)} queries for tuning.")
    
    # Test grid
    k1_vals = [0.8, 1.2, 1.5, 1.8, 2.2]
    b_vals = [0.3, 0.5, 0.6, 0.75, 0.85]
    
    best_ndcg = 0.0
    best_params = None
    
    # We'll run baseline first
    # Baseline k1=1.5, b=0.75
    baseline_ndcgs = []
    for q in eval_queries:
        results = pipeline.search(q["text"], top_k=5)
        retrieved_ids = [doc["id"] for doc in results]
        rel_docs = qrels_dict[q["id"]]
        
        # Calculate NDCG@5
        ret_5 = retrieved_ids[:5]
        dcg_5 = sum(1.0 / math.log2(rank_idx + 2.0) for rank_idx, d in enumerate(ret_5) if d in rel_docs)
        idcg_5 = sum(1.0 / math.log2(rank_idx + 2.0) for rank_idx in range(min(5, len(rel_docs))))
        baseline_ndcgs.append(dcg_5 / idcg_5 if idcg_5 > 0.0 else 0.0)
    
    print(f"Baseline (k1=1.5, b=0.75) NDCG@5: {np.mean(baseline_ndcgs):.4f}")
    
    # Grid search
    for k1 in k1_vals:
        for b in b_vals:
            # Update parameters on all shards
            for shard in pipeline.bm25_retriever.shards:
                shard.k1 = k1
                shard.b = b
                
            ndcgs = []
            for q in eval_queries:
                # Bypass complete search to avoid reranker latency during grid search
                # Just get BM25 results
                cleaned_query = pipeline.preprocessor.clean_query(q["text"])
                corrected_query = pipeline.preprocessor.correct_spelling(cleaned_query)
                bm25_results = pipeline.bm25_retriever.search(corrected_query.split(), top_k=5)
                retrieved_ids = [doc_id for doc_id, _ in bm25_results]
                
                rel_docs = qrels_dict[q["id"]]
                dcg_5 = sum(1.0 / math.log2(rank_idx + 2.0) for rank_idx, d in enumerate(retrieved_ids) if d in rel_docs)
                idcg_5 = sum(1.0 / math.log2(rank_idx + 2.0) for rank_idx in range(min(5, len(rel_docs))))
                ndcgs.append(dcg_5 / idcg_5 if idcg_5 > 0.0 else 0.0)
                
            mean_ndcg = np.mean(ndcgs)
            print(f"Tuning - k1={k1}, b={b} -> BM25 NDCG@5: {mean_ndcg:.4f}")
            
            if mean_ndcg > best_ndcg:
                best_ndcg = mean_ndcg
                best_params = (k1, b)
                
    print(f"Best BM25 params: k1={best_params[0]}, b={best_params[1]} with NDCG@5: {best_ndcg:.4f}")

if __name__ == "__main__":
    main()
