import os
import math
import gc
import json
import heapq
import joblib
import numpy as np
from collections import Counter
from src.data_loader import get_batch_docs
from src.preprocessing import tokenize_text

class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idfs = None
        self.doc_lengths = None
        self.doc_ids = []
        self.avg_dl = 0.0
        self.inverted_index = {}  # vocab_idx -> (doc_indices_np, tfs_np)
        self.corpus_size = 0
        self.index_timestamp = None
        
    def index(self, corpus_dataset, start_idx=0, end_idx=None, shared_stats=None):
        if end_idx is None:
            end_idx = len(corpus_dataset)
        num_docs = end_idx - start_idx
        self.corpus_size = num_docs
        
        print(f"Building vocabulary and statistics for {num_docs} documents...")
        print(f"Using BM25 parameters: k1={self.k1}, b={self.b}")

        doc_lengths = []
        self.doc_ids = []
        sub_batch_size = 50000

        if shared_stats is None:
            term_df = Counter()
            for sub_start in range(start_idx, end_idx, sub_batch_size):
                sub_end = min(sub_start + sub_batch_size, end_idx)
                sub_docs = get_batch_docs(corpus_dataset, sub_start, sub_end)

                for doc in sub_docs:
                    self.doc_ids.append(doc["_id"])
                    title_text = doc.get("title", "") or ""
                    body_text = doc.get("text", "") or ""
                    title_tokens = tokenize_text(title_text)
                    body_tokens = tokenize_text(body_text)
                    tokens = title_tokens + body_tokens
                    doc_lengths.append(len(tokens))
                    for term in set(tokens):
                        term_df[term] += 1

            self.doc_lengths = np.array(doc_lengths, dtype=np.int32)
            self.avg_dl = float(np.mean(self.doc_lengths)) if len(self.doc_lengths) > 0 else 0.0
            self.vocab, self.idfs = self._build_vocab_and_idfs(term_df, num_docs)
        else:
            self.vocab = shared_stats["vocab"]
            self.idfs = shared_stats["idfs"]
            self.avg_dl = shared_stats["avg_dl"]

            for sub_start in range(start_idx, end_idx, sub_batch_size):
                sub_end = min(sub_start + sub_batch_size, end_idx)
                sub_docs = get_batch_docs(corpus_dataset, sub_start, sub_end)

                for doc in sub_docs:
                    self.doc_ids.append(doc["_id"])
                    title_text = doc.get("title", "") or ""
                    body_text = doc.get("text", "") or ""
                    title_tokens = tokenize_text(title_text)
                    body_tokens = tokenize_text(body_text)
                    doc_lengths.append(len(title_tokens) + len(body_tokens))

            self.doc_lengths = np.array(doc_lengths, dtype=np.int32)
        
        import time
        self.index_timestamp = time.time()
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Average doc length: {self.avg_dl:.2f}")
        
        # Build inverted index posting lists
        posting_lists_docs = [[] for _ in range(len(self.vocab))]
        posting_lists_tfs = [[] for _ in range(len(self.vocab))]
        
        for sub_start in range(start_idx, end_idx, sub_batch_size):
            sub_end = min(sub_start + sub_batch_size, end_idx)
            sub_docs = get_batch_docs(corpus_dataset, sub_start, sub_end)
            
            base_doc_idx = sub_start - start_idx
            for i, doc in enumerate(sub_docs):
                doc_idx = base_doc_idx + i
                title_text = doc.get("title", "") or ""
                body_text = doc.get("text", "") or ""
                title_tokens = tokenize_text(title_text)
                body_tokens = tokenize_text(body_text)
                word_counts = Counter(body_tokens)
                for token in title_tokens:
                    word_counts[token] += 2
                for word, count in word_counts.items():
                    if word in self.vocab:
                        term_idx = self.vocab[word]
                        posting_lists_docs[term_idx].append(doc_idx)
                        posting_lists_tfs[term_idx].append(count)
                        
        # Convert posting lists to numpy arrays
        self.inverted_index = {}
        for term_idx in range(len(self.vocab)):
            self.inverted_index[term_idx] = (
                np.array(posting_lists_docs[term_idx], dtype=np.int32),
                np.array(posting_lists_tfs[term_idx], dtype=np.float32)
            )
                        
        gc.collect()

    @staticmethod
    def _build_vocab_and_idfs(term_df, num_docs):
        min_df = min(max(2, int(0.0001 * num_docs)), 100) if num_docs >= 10000 else (2 if num_docs >= 10 else 1)
        max_df = int(0.7 * num_docs) if num_docs >= 10 else num_docs

        print(f"Vocabulary pruning: min_df={min_df}, max_df={max_df}")

        vocab = {}
        temp_idfs = {}

        for term, df in term_df.items():
            if min_df <= df <= max_df:
                vocab[term] = len(vocab)
                idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
                temp_idfs[vocab[term]] = idf

        idfs = np.zeros(len(vocab), dtype=np.float32)
        for idx, idf in temp_idfs.items():
            idfs[idx] = idf

        return vocab, idfs

    def search(self, query_tokens, top_k=100, doc_mask=None):
        if not query_tokens:
            return []
        
        # Clean and deduplicate query tokens
        unique_tokens = list(set(t.lower().strip() for t in query_tokens if t and len(t) > 0))
        if not unique_tokens:
            return []

        candidate_scores = {}
        term_count = 0
        
        for token in unique_tokens:
            if token not in self.vocab:
                continue
            term_count += 1
            term_idx = self.vocab[token]
            idf = self.idfs[term_idx]
            doc_indices, tfs = self.inverted_index[term_idx]
            
            # Prevent division by very small numbers
            if len(doc_indices) == 0:
                continue
                
            lengths = self.doc_lengths[doc_indices]
            safe_avg_dl = max(self.avg_dl, 1.0)  # Prevent division by zero
            denom = tfs + self.k1 * (1.0 - self.b + self.b * lengths / safe_avg_dl)
            tf_scores = tfs * (self.k1 + 1.0) / np.maximum(denom, 0.001)  # Prevent division by zero
            
            for doc_idx, tf_score in zip(doc_indices, tf_scores):
                candidate_scores[doc_idx] = candidate_scores.get(doc_idx, 0.0) + idf * tf_score

        if not candidate_scores:
            return []

        # Apply document mask filter if provided
        if doc_mask is not None:
            mask_set = set(doc_mask)
            candidate_scores = {
                doc_idx: score
                for doc_idx, score in candidate_scores.items()
                if doc_idx < len(self.doc_ids) and self.doc_ids[doc_idx] in mask_set
            }
            if not candidate_scores:
                return []

        # Efficient top-K extraction
        items = list(candidate_scores.items())
        if len(items) > top_k:
            top_items = heapq.nlargest(top_k, items, key=lambda x: x[1])
        else:
            top_items = sorted(items, key=lambda x: x[1], reverse=True)

        return [(self.doc_ids[doc_idx], float(score)) for doc_idx, score in top_items if doc_idx < len(self.doc_ids)]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "vocab": self.vocab,
            "idfs": self.idfs,
            "doc_lengths": self.doc_lengths,
            "doc_ids": self.doc_ids,
            "avg_dl": self.avg_dl,
            "inverted_index": self.inverted_index,
            "k1": self.k1,
            "b": self.b
        }, os.path.join(path, "bm25_index.joblib"))

    def load(self, path):
        data = joblib.load(os.path.join(path, "bm25_index.joblib"))
        self.vocab = data["vocab"]
        self.idfs = data["idfs"]
        self.doc_lengths = data["doc_lengths"]
        self.doc_ids = data["doc_ids"]
        self.avg_dl = data["avg_dl"]
        self.inverted_index = data["inverted_index"]
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)

class ShardedBM25:
    def __init__(self, index_dir, shard_size=1000000, k1=1.5, b=0.75):
        self.index_dir = index_dir
        self.shard_size = shard_size
        self.k1 = k1
        self.b = b
        self.shards = []
        
    def index(self, corpus_dataset):
        num_docs = len(corpus_dataset)
        num_shards = math.ceil(num_docs / self.shard_size)
        print(f"Indexing corpus of size {num_docs} with {num_shards} shards...")

        shared_stats = self._build_shared_stats(corpus_dataset)
        
        for s in range(num_shards):
            start_idx = s * self.shard_size
            end_idx = min(start_idx + self.shard_size, num_docs)
            print(f"Indexing shard {s+1}/{num_shards} (docs {start_idx} to {end_idx})...")
            
            shard_index = BM25Index(k1=self.k1, b=self.b)
            shard_index.index(corpus_dataset, start_idx=start_idx, end_idx=end_idx, shared_stats=shared_stats)
            
            shard_path = os.path.join(self.index_dir, f"shard_{s}")
            shard_index.save(shard_path)
            self.shards.append(shard_index)
            
            gc.collect()

    def _build_shared_stats(self, corpus_dataset):
        num_docs = len(corpus_dataset)
        term_df = Counter()
        doc_lengths = []
        batch_size = 50000

        print(f"Building global vocabulary and statistics for {num_docs} documents...")
        for start_idx in range(0, num_docs, batch_size):
            end_idx = min(start_idx + batch_size, num_docs)
            batch_docs = get_batch_docs(corpus_dataset, start_idx, end_idx)

            for doc in batch_docs:
                title_text = doc.get("title", "") or ""
                body_text = doc.get("text", "") or ""
                title_tokens = tokenize_text(title_text)
                body_tokens = tokenize_text(body_text)
                tokens = title_tokens + body_tokens
                doc_lengths.append(len(tokens))
                for term in set(tokens):
                    term_df[term] += 1

            print(f"Processed global stats for documents {start_idx} to {end_idx}...")

        avg_dl = float(np.mean(np.array(doc_lengths, dtype=np.int32))) if doc_lengths else 0.0
        vocab, idfs = BM25Index._build_vocab_and_idfs(term_df, num_docs)
        print(f"Global vocabulary size: {len(vocab)}")
        print(f"Global average doc length: {avg_dl:.2f}")

        return {
            "vocab": vocab,
            "idfs": idfs,
            "avg_dl": avg_dl
        }

    def load(self):
        self.shards = []
        s = 0
        while True:
            shard_path = os.path.join(self.index_dir, f"shard_{s}")
            if os.path.exists(os.path.join(shard_path, "bm25_index.joblib")):
                shard_index = BM25Index(k1=self.k1, b=self.b)
                shard_index.load(shard_path)
                self.shards.append(shard_index)
                s += 1
            else:
                break
        print(f"Loaded {len(self.shards)} BM25 shards.")
        
    def search(self, query_tokens, top_k=100, doc_mask=None):
        if not self.shards:
            return []
            
        per_shard_k = min(top_k * 3, 2000)  # Get more candidates for better merging
        results_per_shard = [
            shard.search(query_tokens, top_k=per_shard_k, doc_mask=doc_mask)
            for shard in self.shards
        ]
        
        # Merge results from all shards with score normalization
        merged_results = {}
        for shard_res in results_per_shard:
            for doc_id, score in shard_res:
                if doc_id in merged_results:
                    merged_results[doc_id] = max(merged_results[doc_id], score)  # Take max score
                else:
                    merged_results[doc_id] = score
                
        sorted_results = sorted(merged_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
