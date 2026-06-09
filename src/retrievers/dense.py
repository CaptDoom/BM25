import os
import math
import gc
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from src.data_loader import get_batch_docs

class DenseIndex:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", index_type="FlatL2"):
        self.model_name = model_name
        self.index_type = index_type
        self.model = None
        self.faiss_index = None
        self.doc_ids = []
        
    def init_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            
    def index(self, corpus_dataset, start_idx=0, end_idx=None):
        self.init_model()
        if end_idx is None:
            end_idx = len(corpus_dataset)
        num_docs = end_idx - start_idx
        print(f"Generating embeddings for {num_docs} documents...")
        
        self.doc_ids = []
        self.faiss_index = None
        
        sub_batch_size = 50000
        for sub_start in range(start_idx, end_idx, sub_batch_size):
            sub_end = min(sub_start + sub_batch_size, end_idx)
            sub_docs = get_batch_docs(corpus_dataset, sub_start, sub_end)
            
            texts = []
            for doc in sub_docs:
                self.doc_ids.append(doc["_id"])
                texts.append((doc.get("title", "") or "") + " " + (doc.get("text", "") or ""))
                
            embeddings = self.model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
            dimension = embeddings.shape[1]
            
            if self.faiss_index is None:
                print(f"Building FAISS {self.index_type} index...")
                if self.index_type == "FlatL2":
                    self.faiss_index = faiss.IndexFlatL2(dimension)
                elif self.index_type == "HNSW":
                    self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
                else:
                    self.faiss_index = faiss.IndexFlatIP(dimension)
                    
            self.faiss_index.add(embeddings)
            
            del sub_docs, texts, embeddings
            gc.collect()
            
        print(f"Indexed {self.faiss_index.ntotal} vectors.")
        
    def search(self, query_embedding, top_k=100, doc_mask=None):
        distances, indices = self.faiss_index.search(query_embedding, top_k * 2 if doc_mask else top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc_id = self.doc_ids[idx]
            if doc_mask is None or doc_id in doc_mask:
                results.append((doc_id, float(dist)))
                
        return results[:top_k]
        
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.faiss_index, os.path.join(path, "dense_faiss.index"))
        joblib.dump({
            "doc_ids": self.doc_ids,
            "model_name": self.model_name,
            "index_type": self.index_type
        }, os.path.join(path, "dense_metadata.joblib"))
        
    def load(self, path):
        self.faiss_index = faiss.read_index(os.path.join(path, "dense_faiss.index"), faiss.IO_FLAG_MMAP)
        metadata = joblib.load(os.path.join(path, "dense_metadata.joblib"))
        self.doc_ids = metadata["doc_ids"]
        self.model_name = metadata["model_name"]
        self.index_type = metadata["index_type"]
 
class ShardedDense:
    def __init__(self, index_dir, shard_size=1000000, model_name="sentence-transformers/all-MiniLM-L6-v2", index_type="FlatL2"):
        self.index_dir = index_dir
        self.shard_size = shard_size
        self.model_name = model_name
        self.index_type = index_type
        self.shards = []
        self.model = None
        self.query_cache = {}
        
    def init_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            
    def index(self, corpus_dataset):
        num_docs = len(corpus_dataset)
        num_shards = math.ceil(num_docs / self.shard_size)
        print(f"Indexing dense corpus of size {num_docs} with {num_shards} shards...")
        
        for s in range(num_shards):
            start_idx = s * self.shard_size
            end_idx = min(start_idx + self.shard_size, num_docs)
            print(f"Indexing dense shard {s+1}/{num_shards} (docs {start_idx} to {end_idx})...")
            
            shard_index = DenseIndex(model_name=self.model_name, index_type=self.index_type)
            shard_index.model = self.model
            shard_index.index(corpus_dataset, start_idx=start_idx, end_idx=end_idx)
            self.model = shard_index.model
            
            shard_path = os.path.join(self.index_dir, f"dense_shard_{s}")
            shard_index.save(shard_path)
            self.shards.append(shard_index)
            
            gc.collect()

    def load(self):
        self.shards = []
        s = 0
        while True:
            shard_path = os.path.join(self.index_dir, f"dense_shard_{s}")
            if os.path.exists(os.path.join(shard_path, "dense_faiss.index")):
                shard_index = DenseIndex()
                shard_index.load(shard_path)
                self.shards.append(shard_index)
                s += 1
            else:
                break
        print(f"Loaded {len(self.shards)} dense shards.")
        
    def search(self, query_text, top_k=100, doc_mask=None):
        self.init_model()
        
        if query_text in self.query_cache:
            query_embedding = self.query_cache[query_text]
        else:
            query_embedding = self.model.encode([query_text], convert_to_numpy=True)
            self.query_cache[query_text] = query_embedding
            
        results_per_shard = [
            shard.search(query_embedding, top_k=top_k, doc_mask=doc_mask)
            for shard in self.shards
        ]
        
        merged_results = {}
        for shard_res in results_per_shard:
            for doc_id, score in shard_res:
                merged_results[doc_id] = score
                
        # Sort ascending (smaller L2 distance is better)
        sorted_results = sorted(merged_results.items(), key=lambda x: x[1])
        return sorted_results[:top_k]
