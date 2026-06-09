import os
import sqlite3
import json
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def load_corpus(self):
        print(f"Loading corpus for {self.dataset_name}...")
        if self.dataset_name.startswith("BeIR/"):
            return load_dataset(self.dataset_name, "corpus", split="corpus")
        else:
            return load_dataset(self.dataset_name, split="train")

    def load_queries(self):
        print(f"Loading queries for {self.dataset_name}...")
        if self.dataset_name.startswith("BeIR/"):
            return load_dataset(self.dataset_name, "queries", split="queries")
        else:
            return load_dataset(self.dataset_name, split="queries")

    def load_qrels(self):
        print(f"Loading qrels for {self.dataset_name}...")
        if self.dataset_name.startswith("BeIR/"):
            qrels_dataset_name = f"{self.dataset_name}-qrels"
            for split in ["test", "validation", "dev", "train"]:
                try:
                    ds = load_dataset(qrels_dataset_name, split=split)
                    print(f"Loaded qrels split '{split}' from {qrels_dataset_name}")
                    return ds
                except Exception:
                    continue
            raise ValueError(f"Could not find qrels split for {qrels_dataset_name}")
        else:
            return load_dataset(self.dataset_name, split="qrels")

class CorpusDBHelper:
    def __init__(self, db_path):
        self.db_path = db_path
        
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                text TEXT,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()
        
    def insert_documents(self, documents):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        data = []
        for doc in documents:
            meta_str = json.dumps(doc.get("metadata", {})) if "metadata" in doc else "{}"
            data.append((doc["_id"], doc.get("title", ""), doc.get("text", ""), meta_str))
            
        cursor.executemany("""
            INSERT OR REPLACE INTO documents (doc_id, title, text, metadata)
            VALUES (?, ?, ?, ?)
        """, data)
        conn.commit()
        conn.close()
        
    def get_documents(self, doc_ids):
        if not doc_ids:
            return {}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        results = {}
        for i in range(0, len(doc_ids), 500):
            chunk = doc_ids[i:i+500]
            placeholders = ",".join(["?"] * len(chunk))
            cursor.execute(f"SELECT doc_id, title, text, metadata FROM documents WHERE doc_id IN ({placeholders})", chunk)
            for doc_id, title, text, meta_str in cursor.fetchall():
                try:
                    meta = json.loads(meta_str)
                except Exception:
                    meta = {}
                results[doc_id] = {
                    "_id": doc_id,
                    "title": title,
                    "text": text,
                    "metadata": meta
                }
        conn.close()
        return results

    def get_all_doc_ids_matching_filter(self, parsed_filter):
        if not parsed_filter:
            return None
        field, op, val = parsed_filter
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, metadata FROM documents")
        
        matching_ids = set()
        
        for doc_id, meta_str in cursor.fetchall():
            try:
                meta = json.loads(meta_str)
            except Exception:
                meta = {}
            if field in meta:
                doc_val = meta[field]
                if op == "==" and doc_val == val:
                    matching_ids.add(doc_id)
                elif op == "!=" and doc_val != val:
                    matching_ids.add(doc_id)
                elif op == ">=" and doc_val >= val:
                    matching_ids.add(doc_id)
                elif op == "<=" and doc_val <= val:
                    matching_ids.add(doc_id)
                elif op == ">" and doc_val > val:
                    matching_ids.add(doc_id)
                elif op == "<" and doc_val < val:
                    matching_ids.add(doc_id)
                    
        conn.close()
        return matching_ids

def get_batch_docs(dataset, start, end):
    slice_data = dataset[start:end]
    if isinstance(slice_data, list):
        return slice_data
    elif isinstance(slice_data, dict):
        docs = []
        keys = list(slice_data.keys())
        if keys:
            num_items = len(slice_data[keys[0]])
            for i in range(num_items):
                docs.append({k: slice_data[k][i] for k in keys})
        return docs
    else:
        docs = []
        for i in range(start, end):
            docs.append(dataset[i])
        return docs
