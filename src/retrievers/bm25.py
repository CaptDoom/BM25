import os
import gc
import sqlite3
import joblib
import struct
import ctypes
import re
import numpy as np
import bm25s
from src.preprocessing import tokenize_text

try:
    import pyroaring
except ImportError:
    pyroaring = None

def build_metadata_and_bitmaps(index_dir, dataset_name):
    if pyroaring is None:
        print("[Warning] pyroaring is not installed. Skipping Roaring bitmap generation.")
        return False
        
    db_path = os.path.join(index_dir, "corpus.db")
    if not os.path.exists(db_path):
        db_path = os.path.abspath(os.path.join(index_dir, "../corpus.db"))
        if not os.path.exists(db_path):
            db_path = os.path.abspath(os.path.join(index_dir, "../../corpus.db"))
            if not os.path.exists(db_path):
                print(f"[Warning] No corpus.db found for {index_dir}. Cannot build metadata.")
                return False
                
    print(f"Generating derived metadata and Roaring bitmaps from {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_index, title, text, metadata FROM documents ORDER BY doc_index ASC")
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"[Warning] Error querying SQLite database: {e}")
        return False
        
    num_docs = len(rows)
    if num_docs == 0:
        print("[Warning] No documents found in database.")
        return False
        
    # Initialize bitmaps
    bitmaps = {
        "ds_0": pyroaring.BitMap(), "ds_1": pyroaring.BitMap(), "ds_2": pyroaring.BitMap(),
        "ds_3": pyroaring.BitMap(), "ds_4": pyroaring.BitMap(), "ds_5": pyroaring.BitMap(),
        "has_title_0": pyroaring.BitMap(), "has_title_1": pyroaring.BitMap(),
        "len_short": pyroaring.BitMap(), "len_medium": pyroaring.BitMap(), "len_long": pyroaring.BitMap(),
        "has_code_0": pyroaring.BitMap(), "has_code_1": pyroaring.BitMap()
    }
    
    import json
    
    # Dataset ID mapping
    ds_lower = dataset_name.lower()
    if "fever" in ds_lower:
        default_ds_id = 0
    elif "quora" in ds_lower:
        default_ds_id = 1
    elif "scidocs" in ds_lower:
        default_ds_id = 2
    elif "fiqa" in ds_lower:
        default_ds_id = 3
    elif "cqadupstack" in ds_lower:
        default_ds_id = 4
    else:
        default_ds_id = 5
        
    metadata_bin_data = bytearray()
    
    for row in rows:
        doc_index, title, text, meta_str = row
        title = title or ""
        text = text or ""
        combined = title + " " + text
        
        length = len(combined)
        has_title = len(title.strip()) > 0
        
        dataset_id = default_ds_id
        if meta_str:
            try:
                meta = json.loads(meta_str)
                ds_val = meta.get("dataset", "")
                if isinstance(ds_val, str) and ds_val:
                    ds_val_lower = ds_val.lower()
                    if "fever" in ds_val_lower:
                        dataset_id = 0
                    elif "quora" in ds_val_lower:
                        dataset_id = 1
                    elif "scidocs" in ds_val_lower:
                        dataset_id = 2
                    elif "fiqa" in ds_val_lower:
                        dataset_id = 3
                    elif "cqadupstack" in ds_val_lower:
                        dataset_id = 4
            except Exception:
                pass
        
        has_code = False
        if "``" in combined or "`" in combined or "std::" in combined:
            has_code = True
            
        bitmaps[f"ds_{dataset_id}"].add(doc_index)
        bitmaps[f"has_title_{1 if has_title else 0}"].add(doc_index)
        
        if length < 500:
            bitmaps["len_short"].add(doc_index)
        elif length <= 2000:
            bitmaps["len_medium"].add(doc_index)
        else:
            bitmaps["len_long"].add(doc_index)
            
        bitmaps[f"has_code_{1 if has_code else 0}"].add(doc_index)
        
        # Flags packing: bit 0: has_title, bit 1: has_code
        flags = 0
        if has_title:
            flags |= 1
        if has_code:
            flags |= 2
            
        metadata_bin_data.extend(struct.pack('<IHB', length, flags, dataset_id))
        
    os.makedirs(index_dir, exist_ok=True)
    
    # Write metadata array file
    with open(os.path.join(index_dir, "doc_metadata.bin"), "wb") as f:
        f.write(metadata_bin_data)
        
    # Write bitmap index and names
    names = list(bitmaps.keys())
    serialized_bitmaps = [bitmaps[name].serialize() for name in names]
    num_bitmaps = len(names)
    
    header_size = 4 + 8 * num_bitmaps
    offsets = []
    current_offset = header_size
    for sb in serialized_bitmaps:
        offsets.append(current_offset)
        current_offset += len(sb)
        
    with open(os.path.join(index_dir, "bitmaps.bin"), "wb") as f:
        f.write(struct.pack('<I', num_bitmaps))
        for offset in offsets:
            f.write(struct.pack('<Q', offset))
        for sb in serialized_bitmaps:
            f.write(sb)
            
    with open(os.path.join(index_dir, "bitmaps.txt"), "w") as f:
        for name in names:
            f.write(name + "\n")
            
    print(f"Metadata and bitmaps successfully saved to {index_dir}")
    return True

class CppBM25Engine:
    def __init__(self):
        self.lib = None
        self.engine_ptr = None
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        dll_path = os.path.abspath(os.path.join(curr_dir, "..", "cpp", "bm25_score.dll"))
        
        if not os.path.exists(dll_path):
            print(f"[Warning] C++ shared library not found at {dll_path}.")
            return
            
        try:
            self.lib = ctypes.CDLL(dll_path)
            
            self.lib.create_engine.argtypes = [
                ctypes.c_char_p, # indices_path
                ctypes.c_char_p, # indptr_path
                ctypes.c_char_p, # data_path
                ctypes.c_char_p, # metadata_bin_path
                ctypes.c_char_p, # bitmaps_bin_path
                ctypes.c_char_p, # bitmaps_txt_path
                ctypes.c_int     # num_docs
            ]
            self.lib.create_engine.restype = ctypes.c_void_p
            
            self.lib.free_engine.argtypes = [ctypes.c_void_p]
            self.lib.free_engine.restype = None
            
            self.lib.search.argtypes = [
                ctypes.c_void_p,                   # engine_ptr
                ctypes.POINTER(ctypes.c_int),      # query_term_ids
                ctypes.c_int,                      # num_terms
                ctypes.POINTER(ctypes.c_char_p),   # filter_names
                ctypes.c_int,                      # num_filters
                ctypes.c_int,                      # top_k
                ctypes.POINTER(ctypes.c_int),      # out_doc_ids
                ctypes.POINTER(ctypes.c_float)     # out_scores
            ]
            self.lib.search.restype = ctypes.c_int
        except Exception as e:
            print(f"[Warning] Failed to initialize C++ signatures: {e}")
            self.lib = None
            
    def load_index(self, shard_path, num_docs):
        if not self.lib:
            return False
            
        indices_path = os.path.join(shard_path, "indices.csc.index.npy").encode('utf-8')
        indptr_path = os.path.join(shard_path, "indptr.csc.index.npy").encode('utf-8')
        data_path = os.path.join(shard_path, "data.csc.index.npy").encode('utf-8')
        
        metadata_bin_path = os.path.join(shard_path, "doc_metadata.bin").encode('utf-8')
        bitmaps_bin_path = os.path.join(shard_path, "bitmaps.bin").encode('utf-8')
        bitmaps_txt_path = os.path.join(shard_path, "bitmaps.txt").encode('utf-8')
        
        try:
            self.engine_ptr = self.lib.create_engine(
                indices_path,
                indptr_path,
                data_path,
                metadata_bin_path,
                bitmaps_bin_path,
                bitmaps_txt_path,
                num_docs
            )
            return self.engine_ptr is not None
        except Exception as e:
            print(f"[Warning] Failed to load index in C++ create_engine: {e}")
            self.engine_ptr = None
            return False
            
    def search(self, query_term_ids, filter_names, top_k):
        if not self.lib or not self.engine_ptr:
            return []
            
        num_terms = len(query_term_ids)
        term_ids_arr = (ctypes.c_int * num_terms)(*query_term_ids)
        
        num_filters = len(filter_names)
        filter_names_encoded = [name.encode('utf-8') for name in filter_names]
        filter_names_arr = (ctypes.c_char_p * num_filters)(*filter_names_encoded) if num_filters > 0 else None
        
        out_doc_ids = (ctypes.c_int * top_k)()
        out_scores = (ctypes.c_float * top_k)()
        
        try:
            count = self.lib.search(
                self.engine_ptr,
                term_ids_arr,
                num_terms,
                filter_names_arr,
                num_filters,
                top_k,
                out_doc_ids,
                out_scores
            )
            
            results = []
            for i in range(count):
                results.append((out_doc_ids[i], out_scores[i]))
            return results
        except Exception as e:
            print(f"[Warning] C++ search failed: {e}")
            return []
            
    def __del__(self):
        if self.lib and self.engine_ptr:
            try:
                self.lib.free_engine(self.engine_ptr)
            except Exception:
                pass

class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.retriever = None
        self.doc_ids = []
        self.doc_lengths = None
        self.vocab = {}
        self.cpp_engine = None
        self.bitmaps = {}
        self.index_dir = None
        
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
        
    def _load_bitmaps_python(self, path):
        self.bitmaps = {}
        if pyroaring is None:
            return
            
        bin_path = os.path.join(path, "bitmaps.bin")
        txt_path = os.path.join(path, "bitmaps.txt")
        if not os.path.exists(bin_path) or not os.path.exists(txt_path):
            return
            
        try:
            with open(txt_path, "r") as f:
                names = [line.strip() for line in f if line.strip()]
                
            with open(bin_path, "rb") as f:
                num_bitmaps = struct.unpack('<I', f.read(4))[0]
                offsets = [struct.unpack('<Q', f.read(8))[0] for _ in range(num_bitmaps)]
                f.seek(0, 2)
                file_size = f.tell()
                for i, name in enumerate(names):
                    f.seek(offsets[i])
                    size = (offsets[i+1] - offsets[i]) if i + 1 < num_bitmaps else (file_size - offsets[i])
                    data = f.read(size)
                    self.bitmaps[name] = pyroaring.BitMap.deserialize(data)
        except Exception as e:
            print(f"[Warning] Failed to load bitmaps in Python: {e}")

    def search(self, query_tokens, top_k=100, doc_mask=None, filter_names=None):
        if self.retriever is None:
            return []
            
        num_docs = len(self.doc_ids)
        top_k = min(top_k, num_docs)
        if top_k <= 0:
            return []
            
        # Try to use C++ engine
        if self.cpp_engine is not None and not doc_mask:
            query_term_ids = [self.vocab[tok] for tok in query_tokens if tok in self.vocab]
            if not query_term_ids:
                return []
            cpp_res = self.cpp_engine.search(query_term_ids, filter_names or [], top_k)
            out = []
            for doc_idx, score in cpp_res:
                if 0 <= doc_idx < len(self.doc_ids):
                    out.append((self.doc_ids[doc_idx], float(score)))
            return out
            
        # Fallback to Python filtering
        if filter_names:
            if not self.bitmaps and self.index_dir:
                self._load_bitmaps_python(self.index_dir)
            allowed_set = None
            for name in filter_names:
                bm = self.bitmaps.get(name)
                if bm is not None:
                    if allowed_set is None:
                        allowed_set = bm.copy()
                    else:
                        allowed_set &= bm
                else:
                    return []
            if allowed_set is None or len(allowed_set) == 0:
                return []
                
            allowed_indices = set(allowed_set)
            if doc_mask is not None:
                doc_mask_indices = set()
                first_elem = next(iter(doc_mask)) if len(doc_mask) > 0 else None
                if first_elem is not None and isinstance(first_elem, (int, np.integer)):
                    doc_mask_indices = set(doc_mask)
                else:
                    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
                    doc_mask_indices = {doc_id_to_idx[d] for d in doc_mask if d in doc_id_to_idx}
                doc_mask = allowed_indices & doc_mask_indices
            else:
                doc_mask = allowed_indices

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
        self.index_dir = path
        os.makedirs(path, exist_ok=True)
        self.retriever.save(path, corpus=self.doc_ids)
        joblib.dump({
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "vocab": self.retriever.vocab_dict
        }, os.path.join(path, "extra_metadata.joblib"))
        
        # Build metadata and bitmaps
        parts = path.replace("\\", "/").split("/")
        dataset_name = "unknown"
        for p in parts:
            if "BeIR_" in p:
                dataset_name = p.replace("BeIR_", "BeIR/")
                break
        build_metadata_and_bitmaps(path, dataset_name)

    def load(self, path):
        self.index_dir = path
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
            
        # Ensure metadata and bitmaps exist
        metadata_bin = os.path.join(path, "doc_metadata.bin")
        if not os.path.exists(metadata_bin):
            parts = path.replace("\\", "/").split("/")
            dataset_name = "unknown"
            for p in parts:
                if "BeIR_" in p:
                    dataset_name = p.replace("BeIR_", "BeIR/")
                    break
            build_metadata_and_bitmaps(path, dataset_name)
            
        # Load C++ engine
        self.cpp_engine = CppBM25Engine()
        loaded = self.cpp_engine.load_index(path, len(self.doc_ids))
        if not loaded:
            print(f"[Info] C++ scoring engine fallback for {path}.")
            self.cpp_engine = None
            self._load_bitmaps_python(path)

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
        self.cpp_engine = None
        self.bitmaps = {}
        
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
        
        self.save(self.index_dir)
        self.vocab = self.retriever.vocab_dict
        self.shards = [self]
        gc.collect()

    def save(self, path):
        self.index_dir = path
        os.makedirs(path, exist_ok=True)
        self.retriever.save(path, corpus=self.doc_ids)
        joblib.dump({
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "vocab": self.retriever.vocab_dict
        }, os.path.join(path, "extra_metadata.joblib"))
        
        # Build metadata and bitmaps
        parts = path.replace("\\", "/").split("/")
        dataset_name = "unknown"
        for p in parts:
            if "BeIR_" in p:
                dataset_name = p.replace("BeIR_", "BeIR/")
                break
        build_metadata_and_bitmaps(path, dataset_name)

    def _load_bitmaps_python(self, path):
        self.bitmaps = {}
        if pyroaring is None:
            return
            
        bin_path = os.path.join(path, "bitmaps.bin")
        txt_path = os.path.join(path, "bitmaps.txt")
        if not os.path.exists(bin_path) or not os.path.exists(txt_path):
            return
            
        try:
            with open(txt_path, "r") as f:
                names = [line.strip() for line in f if line.strip()]
                
            with open(bin_path, "rb") as f:
                num_bitmaps = struct.unpack('<I', f.read(4))[0]
                offsets = [struct.unpack('<Q', f.read(8))[0] for _ in range(num_bitmaps)]
                f.seek(0, 2)
                file_size = f.tell()
                for i, name in enumerate(names):
                    f.seek(offsets[i])
                    size = (offsets[i+1] - offsets[i]) if i + 1 < num_bitmaps else (file_size - offsets[i])
                    data = f.read(size)
                    self.bitmaps[name] = pyroaring.BitMap.deserialize(data)
        except Exception as e:
            print(f"[Warning] Failed to load bitmaps in Python: {e}")

    def load(self):
        # 1. Check if there are sharded directories (e.g. shard_0)
        shard_0_path = os.path.join(self.index_dir, "shard_0")
        if os.path.exists(shard_0_path):
            print(f"Detected sharded index at {self.index_dir}. Loading shards...")
            self.shards = []
            s = 0
            while True:
                shard_path = os.path.join(self.index_dir, f"shard_{s}")
                if os.path.exists(os.path.join(shard_path, "data.csc.index.npy")) or os.path.exists(os.path.join(shard_path, "bm25_index.joblib")):
                    shard_index = BM25Index(k1=self.k1, b=self.b)
                    shard_index.load(shard_path)
                    self.shards.append(shard_index)
                    s += 1
                else:
                    break
            print(f"Loaded {len(self.shards)} BM25 shards.")
            
            self.doc_ids = []
            self.vocab = {}
            for shard in self.shards:
                self.doc_ids.extend(shard.doc_ids)
                self.vocab.update(shard.vocab)
            return

        # 2. Single-index loading case
        bm25s_index_file = os.path.join(self.index_dir, "data.csc.index.npy")
        
        if not os.path.exists(bm25s_index_file):
            db_path = os.path.join(os.path.dirname(self.index_dir), "corpus.db")
            if os.path.exists(db_path):
                if "BeIR_fever" in self.index_dir:
                    print(f"FEVER index files missing at {self.index_dir}. Skipping auto-rebuild to prevent OOM/hang.")
                    return
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
        
        meta_path = os.path.join(self.index_dir, "extra_metadata.joblib")
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.doc_ids = meta["doc_ids"]
            self.doc_lengths = meta["doc_lengths"]
        else:
            self.doc_ids = [doc['text'] for doc in self.retriever.corpus]
            self.doc_lengths = np.ones(len(self.doc_ids), dtype=np.int32) * 100
            
        # Ensure metadata and bitmaps exist
        metadata_bin = os.path.join(self.index_dir, "doc_metadata.bin")
        if not os.path.exists(metadata_bin):
            parts = self.index_dir.replace("\\", "/").split("/")
            dataset_name = "unknown"
            for p in parts:
                if "BeIR_" in p:
                    dataset_name = p.replace("BeIR_", "BeIR/")
                    break
            build_metadata_and_bitmaps(self.index_dir, dataset_name)
            
        # Load C++ engine
        self.cpp_engine = CppBM25Engine()
        loaded = self.cpp_engine.load_index(self.index_dir, len(self.doc_ids))
        if not loaded:
            print(f"[Info] C++ scoring engine fallback for {self.index_dir}.")
            self.cpp_engine = None
            self._load_bitmaps_python(self.index_dir)
            
        self.shards = [self]
        
    def _search_single(self, query_tokens, top_k=100, doc_mask=None, filter_names=None):
        if self.retriever is None:
            return []
            
        num_docs = len(self.doc_ids)
        top_k = min(top_k, num_docs)
        if top_k <= 0:
            return []
            
        if self.cpp_engine is not None and not doc_mask:
            query_term_ids = [self.vocab[tok] for tok in query_tokens if tok in self.vocab]
            if not query_term_ids:
                return []
            cpp_res = self.cpp_engine.search(query_term_ids, filter_names or [], top_k)
            out = []
            for doc_idx, score in cpp_res:
                if 0 <= doc_idx < len(self.doc_ids):
                    out.append((self.doc_ids[doc_idx], float(score)))
            return out
            
        if filter_names:
            if not self.bitmaps and self.index_dir:
                self._load_bitmaps_python(self.index_dir)
            allowed_set = None
            for name in filter_names:
                bm = self.bitmaps.get(name)
                if bm is not None:
                    if allowed_set is None:
                        allowed_set = bm.copy()
                    else:
                        allowed_set &= bm
                else:
                    return []
            if allowed_set is None or len(allowed_set) == 0:
                return []
                
            allowed_indices = set(allowed_set)
            if doc_mask is not None:
                doc_mask_indices = set()
                first_elem = next(iter(doc_mask)) if len(doc_mask) > 0 else None
                if first_elem is not None and isinstance(first_elem, (int, np.integer)):
                    doc_mask_indices = set(doc_mask)
                else:
                    if not hasattr(self, "_doc_id_to_idx"):
                        self._doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
                    doc_mask_indices = {self._doc_id_to_idx[d] for d in doc_mask if d in self._doc_id_to_idx}
                doc_mask = allowed_indices & doc_mask_indices
            else:
                doc_mask = allowed_indices

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

    def search(self, query_tokens, top_k=100, doc_mask=None, filter_names=None):
        if not self.shards:
            return []
            
        if len(self.shards) == 1 and self.shards[0] == self:
            return self._search_single(query_tokens, top_k=top_k, doc_mask=doc_mask, filter_names=filter_names)
            
        # Sharded search across loaded shards
        results_per_shard = []
        for shard in self.shards:
            res = shard.search(query_tokens, top_k=top_k, doc_mask=doc_mask, filter_names=filter_names)
            results_per_shard.append(res)
            
        # Merge results from all shards
        merged_results = {}
        for shard_res in results_per_shard:
            for doc_id, score in shard_res:
                if doc_id in merged_results:
                    merged_results[doc_id] = max(merged_results[doc_id], score)
                else:
                    merged_results[doc_id] = score
                    
        sorted_results = sorted(merged_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def retrieve(self, query, filter_mask=None, filter_names=None, k=100):
        if isinstance(query, str):
            query_tokens = tokenize_text(query)
        else:
            query_tokens = query
        return self.search(query_tokens, top_k=k, doc_mask=filter_mask, filter_names=filter_names)
