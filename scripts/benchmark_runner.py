import time
import os
import sys
import yaml
import numpy as np

# Resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import RetrievalPipeline

def run_benchmarks():
    print("=== Running AuraAI Performance Benchmarks ===")
    
    config_path = "config/config.yaml"
    index_dir = "src/index/BeIR_scidocs"
    
    if not os.path.exists(index_dir):
        print(f"[Info] Index directory {index_dir} not found. Skipping benchmark execution.")
        return
        
    try:
        pipeline = RetrievalPipeline(config_path, index_dir)
        pipeline.load_indexes()
        
        queries = [
            "machine learning algorithms",
            "deep neural networks",
            "natural language processing",
            "artificial intelligence in health",
            "database indexes and query optimization"
        ]
        
        latencies = []
        # Warmup
        pipeline.search(queries[0], top_k=5)
        
        # Latency Benchmark
        num_runs = 20
        for _ in range(num_runs):
            for query in queries:
                t_start = time.perf_counter()
                pipeline.search(query, top_k=10, use_reranker=False)
                t_end = time.perf_counter()
                latencies.append((t_end - t_start) * 1000.0) # in ms
                
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        throughput = len(queries) * num_runs / (sum(latencies) / 1000.0)
        
        print(f"\n--- Results ---")
        print(f"p50 Latency:     {p50:.2f} ms")
        print(f"p95 Latency:     {p95:.2f} ms")
        print(f"p99 Latency:     {p99:.2f} ms")
        print(f"Throughput:      {throughput:.2f} queries/sec")
        print("=============================================")
        
    except Exception as e:
        print(f"[Error] Benchmark run failed: {e}")

if __name__ == "__main__":
    run_benchmarks()
