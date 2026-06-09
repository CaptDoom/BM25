import os
import argparse
import yaml
from src.pipeline import RetrievalPipeline

def main():
    parser = argparse.ArgumentParser(description="Interactive search CLI for sparse retrieval pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--index_dir", type=str, default="src/index", help="Directory for storing indices")
    parser.add_argument("--dataset", type=str, help="Hugging Face BEIR dataset source, e.g., BeIR/scidocs")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    dataset_name = args.dataset if args.dataset else config.get("dataset_name", "BeIR/scidocs")
    safe_dataset_name = dataset_name.replace("/", "_")
    dataset_index_dir = os.path.join(args.index_dir, safe_dataset_name)
    
    if not os.path.exists(os.path.join(dataset_index_dir, "corpus.db")):
        print(f"Index does not exist for dataset '{dataset_name}' in '{dataset_index_dir}'. Please index first using evaluate.py.")
        return
        
    print(f"Initializing retrieval pipeline for dataset: {dataset_name}...")
    pipeline = RetrievalPipeline(args.config, dataset_index_dir)
    pipeline.load_indexes()
    print("Pipeline ready! Enter queries below.")
    
    while True:
        try:
            query = input("\nEnter query (or 'quit' to exit): ").strip()
            if not query:
                continue
            if query.lower() == 'quit':
                break
                
            filter_str = input("Enter metadata filter (or press Enter for none, e.g. year == 2020): ").strip()
            filter_str = filter_str if filter_str else None
            
            results = pipeline.search(query, metadata_filter=filter_str, top_k=5)
            
            if not results:
                print("No documents retrieved.")
                continue
                
            print(f"\nTop results for '{query}':")
            for idx, doc in enumerate(results):
                print(f"{idx+1}. ID: {doc['id']} (Reranked Score: {doc['score']:.4f})")
                print(f"   Title: {doc['title']}")
                print(f"   Text:  {doc['text']}")
                print(f"   Meta:  {doc['metadata']}\n")
        except (KeyboardInterrupt, EOFError):
            print()
            break

if __name__ == "__main__":
    main()
