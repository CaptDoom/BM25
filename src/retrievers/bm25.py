import os
import gc
import sqlite3
import joblib
import numpy as np
import bm25s
from src.preprocessing import tokenize_text

class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.retriever = None
        self.doc_ids = []
        self.doc_lengths = None
        self.vocab = {}
        
    def index(self, corpus_dataset, start_idx=0, end_idx=None, shared_stats=None):
        self.doc_ids = []
        texts = []
        for doc in corpus_dataset:
            self.doc_ids.append(doc["_id"])
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            texts.append(title + " " + text)
            
        corpus_tokens = [tokenize_text(t) for t in texts]
        self.doc_lengths = np.array([len(tokens) for tokens in corpus_tokens], dtype=np.int32)
        
        self.retriever = bm25s.BM25(k1=self.k1, b=self.b, corpus=self.doc_ids)
        self.retriever.index(corpus_tokens)
        self.vocab = self.retriever.vocab_dict
        
    def search(self, query_tokens, top_k=100, doc_mask=None):
        if self.retriever is None:
            return []
        
        num_docs = len(self.doc_ids)
        top_k = min(top_k, num_docs)
        if top_k <= 0:
            return []
            
        weight_mask = None
        if doc_mask is not None:
            weight_mask = np.zeros(num_docs, dtype=np.float32)
            
            is_int_mask = False
            if len(doc_mask) > 0:
                first_elem = next(iter(doc_mask))
                if isinstance(first_elem, (int, np.integer)):
                    is_int_mask = True
                    
            if is_int_mask:
                indices = list(doc_mask)
                indices = [idx for idx in indices if 0 <= idx < num_docs]
                weight_mask[indices] = 1.0
            else:
                doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
                indices = [doc_id_to_idx[doc_id] for doc_id in doc_mask if doc_id in doc_id_to_idx]
                weight_mask[indices] = 1.0
                
        results = self.retriever.retrieve([query_tokens], k=top_k, weight_mask=weight_mask)
        out = []
        if len(results.documents) > 0:
            for doc, score in zip(results.documents[0], results.scores[0]):
                if score > 0.0:
                    if isinstance(doc, dict):
                        doc_id = doc.get('text', '')
                    else:
                        doc_id = str(doc)
                    out.append((doc_id, float(score)))
        return out

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.retriever.save(path, corpus=self.doc_ids)
        joblib.dump({
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "vocab": self.retriever.vocab_dict
        }, os.path.join(path, "extra_metadata.joblib"))

    def load(self, path):
        self.retriever = bm25s.BM25.load(path, load_corpus=True)
        self.vocab = self.retriever.vocab_dict
        meta_path = os.path.join(path, "extra_metadata.joblib")
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.doc_ids = meta["doc_ids"]
            self.doc_lengths = meta["doc_lengths"]
        else:
            self.doc_ids = [doc['text'] for doc in self.retriever.corpus]
            self.doc_lengths = np.ones(len(self.doc_ids), dtype=np.int32) * 100

class ShardedBM25:
    def __init__(self, index_dir, shard_size=1000000, k1=1.5, b=0.75):
        self.index_dir = index_dir
        self.shard_size = shard_size
        self.k1 = k1
        self.b = b
        self.retriever = None
        self.doc_ids = []
        self.doc_lengths = None
        self.vocab = {}
        self.shards = []
        
    def index(self, corpus_dataset):
        print(f"Indexing corpus of size {len(corpus_dataset)} using bm25s...")
        self.doc_ids = []
        texts = []
        for doc in corpus_dataset:
            self.doc_ids.append(doc["_id"])
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            texts.append(title + " " + text)
            
        print("Tokenizing corpus...")
        corpus_tokens = [tokenize_text(t) for t in texts]
        self.doc_lengths = np.array([len(tokens) for tokens in corpus_tokens], dtype=np.int32)
        
        print(f"Building bm25s index with k1={self.k1}, b={self.b}...")
        self.retriever = bm25s.BM25(k1=self.k1, b=self.b, corpus=self.doc_ids)
        self.retriever.index(corpus_tokens)
        
        # Save index
        self.save(self.index_dir)
        self.vocab = self.retriever.vocab_dict
        self.shards = [self]
        gc.collect()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.retriever.save(path, corpus=self.doc_ids)
        joblib.dump({
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "vocab": self.retriever.vocab_dict
        }, os.path.join(path, "extra_metadata.joblib"))

    def load(self):
        bm25s_index_file = os.path.join(self.index_dir, "data.csc.index.npy")
        
        # Auto-migration: If bm25s files are missing but SQLite database exists, rebuild the bm25s index automatically
        if not os.path.exists(bm25s_index_file):
            db_path = os.path.join(os.path.dirname(self.index_dir), "corpus.db")
            if os.path.exists(db_path):
                print(f"bm25s index files not found at {self.index_dir}. Rebuilding from SQLite database...")
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT doc_id, title, text FROM documents ORDER BY doc_index ASC")
                    rows = cursor.fetchall()
                    conn.close()
                    if rows:
                        corpus_dataset = [{"_id": r[0], "title": r[1], "text": r[2]} for r in rows]
                        self.index(corpus_dataset)
                        return
                except Exception as e:
                    print(f"[WARNING] Failed to auto-rebuild index from SQLite: {e}")
                    
        print(f"Loading bm25s index from {self.index_dir}...")
        self.retriever = bm25s.BM25.load(self.index_dir, load_corpus=True)
        self.vocab = self.retriever.vocab_dict
        
        # Load extra metadata if exists
        meta_path = os.path.join(self.index_dir, "extra_metadata.joblib")
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.doc_ids = meta["doc_ids"]
            self.doc_lengths = meta["doc_lengths"]
        else:
            # Reconstruct from loaded corpus
            self.doc_ids = [doc['text'] for doc in self.retriever.corpus]
            # Mock doc_lengths if missing
            self.doc_lengths = np.ones(len(self.doc_ids), dtype=np.int32) * 100
            
        self.shards = [self]
        
    def search(self, query_tokens, top_k=100, doc_mask=None):
        if self.retriever is None:
            return []
            
        num_docs = len(self.doc_ids)
        top_k = min(top_k, num_docs)
        if top_k <= 0:
            return []
            
        weight_mask = None
        if doc_mask is not None:
            weight_mask = np.zeros(num_docs, dtype=np.float32)
            
            is_int_mask = False
            if len(doc_mask) > 0:
                first_elem = next(iter(doc_mask))
                if isinstance(first_elem, (int, np.integer)):
                    is_int_mask = True
                    
            if is_int_mask:
                indices = list(doc_mask)
                indices = [idx for idx in indices if 0 <= idx < num_docs]
                weight_mask[indices] = 1.0
            else:
                if not hasattr(self, "_doc_id_to_idx"):
                    self._doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
                indices = [self._doc_id_to_idx[doc_id] for doc_id in doc_mask if doc_id in self._doc_id_to_idx]
                weight_mask[indices] = 1.0
                
        # Run retrieval
        results = self.retriever.retrieve([query_tokens], k=top_k, weight_mask=weight_mask)
        
        out = []
        if len(results.documents) > 0:
            for doc, score in zip(results.documents[0], results.scores[0]):
                if score > 0.0:
                    if isinstance(doc, dict):
                        doc_id = doc.get('text', '')
                    else:
                        doc_id = str(doc)
                    out.append((doc_id, float(score)))
        return out

    def retrieve(self, query, filter_mask=None, k=100):
        if isinstance(query, str):
            query_tokens = tokenize_text(query)
        else:
            query_tokens = query
        return self.search(query_tokens, top_k=k, doc_mask=filter_mask)
