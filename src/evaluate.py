import os
import argparse
import math
import yaml
import numpy as np
from src.data_loader import DataLoader, CorpusDBHelper, get_batch_docs
from src.retrievers.bm25 import ShardedBM25
from src.pipeline import RetrievalPipeline

def build_index(dataset_name, config_path, index_dir):
    print(f"Starting index build for dataset '{dataset_name}' to '{index_dir}'...")
    loader = DataLoader(dataset_name)
    corpus = loader.load_corpus()
    
    # 1. Initialize SQLite Database
    os.makedirs(index_dir, exist_ok=True)
    db_path = os.path.join(index_dir, "corpus.db")
    db_helper = CorpusDBHelper(db_path)
    db_helper.init_db()
    
    # Insert documents in batches to SQLite
    print("Writing documents to SQLite database...")
    batch_size = 50000
    num_docs = len(corpus)
    for i in range(0, num_docs, batch_size):
        end_idx = min(i + batch_size, num_docs)
        batch_docs = get_batch_docs(corpus, i, end_idx)
        db_helper.insert_documents(batch_docs, start_idx=i)
        print(f"Inserted documents {i} to {end_idx}...")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Validate SystemConfig configuration schema
    from src.config_schema import SystemConfig
    SystemConfig(**config)

        
    # 2. Build Sharded BM25 index
    bm25_cfg = config.get("bm25", {})
    bm25_retriever = ShardedBM25(
        index_dir=os.path.join(index_dir, "bm25"),
        shard_size=config.get("sharding", {}).get("shard_size", 1000000),
        k1=bm25_cfg.get("k1", 1.2),
        b=bm25_cfg.get("b", 0.75)
    )
    bm25_retriever.index(corpus)
    
    # 3. Skip Dense Index (Sparse Retrieval Mode)
    print("Skipping Dense Indexing (SPARSE-ONLY)...")
    
    # 4. Save stats.json
    import json
    num_docs = len(corpus)
    vocab = set()
    total_dl = 0.0
    for shard in bm25_retriever.shards:
        vocab.update(shard.vocab.keys())
        total_dl += np.sum(shard.doc_lengths)
    vocab_size = len(vocab)
    avg_dl = float(total_dl / num_docs) if num_docs > 0 else 0.0
    
    stats = {
        "num_docs": num_docs,
        "vocab_size": vocab_size,
        "avg_dl": avg_dl
    }
    with open(os.path.join(index_dir, "stats.json"), "w") as f:
        json.dump(stats, f)
        
    print("Indexing completed successfully!")

def run_evaluation(pipeline, dataset_name):
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
    
    print(f"Number of queries for evaluation: {len(eval_queries)}")
    if not eval_queries:
        print("No evaluation queries found.")
        return
        
    precisions_5, recalls_5, ndcgs_5 = [], [], []
    precisions_10, recalls_10, ndcgs_10 = [], [], []
    
    # Evaluate top 100 queries to speed up execution
    limit = 100 if len(eval_queries) > 100 else len(eval_queries)
    print(f"Running evaluation on top {limit} queries...")
    
    for idx in range(limit):
        q = eval_queries[idx]
        q_id = q["id"]
        q_text = q["text"]
        
        rel_docs = qrels_dict[q_id]
        results = pipeline.search(q_text, top_k=10, use_reranker=True)
        retrieved_ids = [doc["id"] for doc in results]
        
        # Metrics @ 5
        ret_5 = retrieved_ids[:5]
        hits_5 = sum(1 for d in ret_5 if d in rel_docs)
        precisions_5.append(hits_5 / 5.0)
        recalls_5.append(hits_5 / len(rel_docs) if rel_docs else 0.0)
        
        dcg_5 = sum(1.0 / math.log2(rank_idx + 2.0) for rank_idx, d in enumerate(ret_5) if d in rel_docs)
        idcg_5 = sum(1.0 / math.log2(rank_idx + 2.0) for rank_idx in range(min(5, len(rel_docs))))
        ndcgs_5.append(dcg_5 / idcg_5 if idcg_5 > 0.0 else 0.0)
        
        # Metrics @ 10
        hits_10 = sum(1 for d in retrieved_ids if d in rel_docs)
        precisions_10.append(hits_10 / 10.0)
        recalls_10.append(hits_10 / len(rel_docs) if rel_docs else 0.0)
        
        dcg_10 = sum(1.0 / math.log2(rank_idx + 2.0) for rank_idx, d in enumerate(retrieved_ids) if d in rel_docs)
        idcg_10 = sum(1.0 / math.log2(rank_idx + 2.0) for rank_idx in range(min(10, len(rel_docs))))
        ndcgs_10.append(dcg_10 / idcg_10 if idcg_10 > 0.0 else 0.0)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{limit} queries...")
        
    print("\n=== Evaluation Results ===")
    print(f"Precision@5:  {np.mean(precisions_5):.4f}")
    print(f"Recall@5:     {np.mean(recalls_5):.4f}")
    print(f"NDCG@5:       {np.mean(ndcgs_5):.4f}")
    print(f"Precision@10: {np.mean(precisions_10):.4f}")
    print(f"Recall@10:    {np.mean(recalls_10):.4f}")
    print(f"NDCG@10:      {np.mean(ndcgs_10):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Build and evaluate sparse retrieval pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--index_dir", type=str, default="src/index", help="Directory for storing indices")
    parser.add_argument("--dataset", type=str, help="Hugging Face BEIR dataset source, e.g., BeIR/scidocs")
    parser.add_argument("--index_only", action="store_true", help="Only index the dataset")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    from src.config_schema import SystemConfig
    SystemConfig(**config)

        
    dataset_name = args.dataset if args.dataset else config.get("dataset_name", "BeIR/scidocs")
    # Clean dataset name for path safely
    safe_dataset_name = dataset_name.replace("/", "_")
    dataset_index_dir = os.path.join(args.index_dir, safe_dataset_name)
    
    # Check if index exists or rebuild requested
    index_exists = os.path.exists(os.path.join(dataset_index_dir, "corpus.db"))
    
    if not args.evaluate_only and (args.index_only or not index_exists):
        build_index(dataset_name, args.config, dataset_index_dir)
        
    if not args.index_only:
        pipeline = RetrievalPipeline(args.config, dataset_index_dir)
        run_evaluation(pipeline, dataset_name)

if __name__ == "__main__":
    main()
