import os
import sys
import time
import traceback

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate import build_index

DATASETS = [
    "BeIR/scidocs",
    "BeIR/fiqa",
    "BeIR/quora",
    "BeIR/fever",
    "BeIR/msmarco"
]

CONFIG_PATH = "config/config.yaml"
INDEX_ROOT_DIR = "src/index"

def main():
    print("Starting indexing pipeline for all datasets...")
    for dataset in DATASETS:
        safe_name = dataset.replace("/", "_")
        index_dir = os.path.join(INDEX_ROOT_DIR, safe_name)
        
        # Check if already indexed
        if os.path.exists(os.path.join(index_dir, "corpus.db")) and os.path.exists(os.path.join(index_dir, "stats.json")):
            print(f"\nIndex for {dataset} already exists at {index_dir}. Skipping...")
            continue
            
        print(f"\n==================================================")
        print(f"Indexing dataset: {dataset}")
        print(f"==================================================")
        
        t_start = time.time()
        try:
            build_index(dataset, CONFIG_PATH, index_dir)
            t_elapsed = time.time() - t_start
            print(f"Successfully indexed {dataset} in {t_elapsed:.2f} seconds.")
        except Exception as e:
            print(f"FAILED to index {dataset}!")
            traceback.print_exc()
            
    print("\nAll datasets processed!")

if __name__ == "__main__":
    main()
