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
    print(f"Checking for corpus.db in index_dir: {db_path} -> {os.path.exists(db_path)}")
    if not os.path.exists(db_path):
        # Check parent
        db_path = os.path.abspath(os.path.join(index_dir, "..", "corpus.db"))
        print(f"Checking for corpus.db in parent: {db_path} -> {os.path.exists(db_path)}")
        if not os.path.exists(db_path):
            # Check grandparent
            db_path = os.path.abspath(os.path.join(index_dir, "..", "..", "corpus.db"))
            print(f"Checking for corpus.db in grandparent: {db_path} -> {os.path.exists(db_path)}")
            if not os.path.exists(db_path):
                # Check siblings if within a staging context
                db_path = os.path.abspath(os.path.join(os.path.dirname(os.path.normpath(index_dir)), "corpus.db"))
                print(f"Checking for corpus.db in sibling: {db_path} -> {os.path.exists(db_path)}")
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
        
        # Build integrity: validate row values and handle nulls/mismatches
        if doc_index is None:
            continue
            
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
                ctypes.c_void_p,                   # doc_mask_data
                ctypes.c_int,                      # doc_mask_size
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
            
        abs_shard_path = os.path.abspath(shard_path)
        indices_path = os.path.join(abs_shard_path, "indices.csc.index.npy").encode('utf-8')
        indptr_path = os.path.join(abs_shard_path, "indptr.csc.index.npy").encode('utf-8')
        data_path = os.path.join(abs_shard_path, "data.csc.index.npy").encode('utf-8')
        
        metadata_bin_path = os.path.join(abs_shard_path, "doc_metadata.bin").encode('utf-8')
        bitmaps_bin_path = os.path.join(abs_shard_path, "bitmaps.bin").encode('utf-8')
        bitmaps_txt_path = os.path.join(abs_shard_path, "bitmaps.txt").encode('utf-8')
        
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

            
    def search(self, query_term_ids, filter_names, top_k, doc_mask_buffer=None, doc_mask_size=0):
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
                doc_mask_buffer,
                doc_mask_size,
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
        self.doc_id_to_idx = {}
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
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        
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

        doc_mask = self._normalize_doc_mask(doc_mask)
        if doc_mask is not None and len(doc_mask) == 0:
            return []

        doc_mask_buffer = None
        doc_mask_size = 0
        if doc_mask is not None and pyroaring is not None:
            doc_mask_bitmap = pyroaring.BitMap(sorted(doc_mask))
            doc_mask_bytes = doc_mask_bitmap.serialize()
            if doc_mask_bytes:
                doc_mask_buffer = ctypes.create_string_buffer(doc_mask_bytes)
                doc_mask_size = len(doc_mask_bytes)
            else:
                return []
            
        # Try to use C++ engine
        if self.cpp_engine is not None:
            query_term_ids = [self.vocab[tok] for tok in query_tokens if tok in self.vocab]
            if not query_term_ids:
                return []
            if doc_mask is not None and pyroaring is None:
                # Fall back to Python if we cannot serialize the metadata mask.
                pass
            else:
                cpp_res = self.cpp_engine.search(
                    query_term_ids,
                    filter_names or [],
                    top_k,
                    ctypes.cast(doc_mask_buffer, ctypes.c_void_p) if doc_mask_buffer is not None else None,
                    doc_mask_size,
                )
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
                doc_mask = allowed_indices & doc_mask
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
                indices = [idx for idx in doc_mask if 0 <= idx < num_docs]
                weight_mask[indices] = 1.0
            else:
                indices = [self.doc_id_to_idx[doc_id] for doc_id in doc_mask if doc_id in self.doc_id_to_idx]
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

    def _normalize_doc_mask(self, doc_mask):
        if doc_mask is None:
            return None
        if isinstance(doc_mask, np.ndarray):
            if doc_mask.size == 0:
                return set()
            if np.issubdtype(doc_mask.dtype, np.integer):
                return {int(idx) for idx in doc_mask.tolist()}
            return {int(idx) for idx in doc_mask.tolist() if isinstance(idx, (int, np.integer))}

        if isinstance(doc_mask, (set, frozenset, list, tuple)):
            if not doc_mask:
                return set()
            first_elem = next(iter(doc_mask))
            if isinstance(first_elem, (int, np.integer)):
                return {int(idx) for idx in doc_mask if isinstance(idx, (int, np.integer))}
            return {self.doc_id_to_idx[doc_id] for doc_id in doc_mask if doc_id in self.doc_id_to_idx}

        return None

    def save(self, path):
        import shutil
        import hashlib
        import json
        
        # 1. Create a staging directory under index_dir
        parent_dir = os.path.dirname(os.path.normpath(path))
        base_name = os.path.basename(os.path.normpath(path))
        staging_path = os.path.join(parent_dir, f"{base_name}_staging")
        if os.path.exists(staging_path):
            shutil.rmtree(staging_path)
        os.makedirs(staging_path, exist_ok=True)
        
        # Save indices inside staging
        self.retriever.save(staging_path, corpus=self.doc_ids)
        joblib.dump({
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "vocab": self.retriever.vocab_dict
        }, os.path.join(staging_path, "extra_metadata.joblib"))
        
        # Copy corpus.db to staging if it exists in path (needed for metadata generation in staging)
        db_source = os.path.join(path, "corpus.db")
        if os.path.exists(db_source):
            shutil.copy2(db_source, os.path.join(staging_path, "corpus.db"))
            
        parts = path.replace("\\", "/").split("/")
        dataset_name = "unknown"
        for p in parts:
            if "BeIR_" in p:
                dataset_name = p.replace("BeIR_", "BeIR/")
                break
        build_metadata_and_bitmaps(staging_path, dataset_name)


        
        # Write cryptographic manifest with SHA-256
        manifest = {
            "schema_version": "1.0",
            "item_count": len(self.doc_ids),
            "checksums": {}
        }
        for file in os.listdir(staging_path):
            file_path = os.path.join(staging_path, file)
            if os.path.isfile(file_path):
                hasher = hashlib.sha256()
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
                manifest["checksums"][file] = hasher.hexdigest()
                
        with open(os.path.join(staging_path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
            
        # Atomic Swap
        if os.path.exists(path):
            shutil.rmtree(path)
        os.rename(staging_path, path)
        self.index_dir = path

    def load(self, path):
        import hashlib
        import json
        self.index_dir = path
        
        # Verify Manifest Checksums on load
        manifest_path = os.path.join(path, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                checksums = manifest.get("checksums", {})
                for file, expected_sha in checksums.items():
                    file_path = os.path.join(path, file)
                    if os.path.exists(file_path):
                        hasher = hashlib.sha256()
                        with open(file_path, "rb") as f:
                            while chunk := f.read(8192):
                                hasher.update(chunk)
                        if hasher.hexdigest() != expected_sha:
                            print(f"[Warning] Index file checksum mismatch for {file}!")
            except Exception as e:
                print(f"[Warning] Failed to verify index manifest checksums: {e}")
        
        self.retriever = bm25s.BM25.load(path, load_corpus=True, mmap=False)
        self.vocab = self.retriever.vocab_dict
        meta_path = os.path.join(path, "extra_metadata.joblib")
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.doc_ids = meta["doc_ids"]
            self.doc_lengths = meta["doc_lengths"]
        else:
            self.doc_ids = []
            for idx, doc in enumerate(self.retriever.corpus):
                if isinstance(doc, dict):
                    self.doc_ids.append(str(doc.get("_id", doc.get("id", idx))))
                else:
                    self.doc_ids.append(str(idx))
            self.doc_lengths = np.ones(len(self.doc_ids), dtype=np.int32) * 100
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
            
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
            
        # Load C++ engine with relative resolution
        self.cpp_engine = CppBM25Engine()
        loaded = self.cpp_engine.load_index(path, len(self.doc_ids))
        if not loaded:
            print(f"[Info] C++ scoring engine fallback for {path}.")
            self.cpp_engine = None
            self._load_bitmaps_python(path)
            
        # Cold-start Viability Health Check
        try:
            health_check_tokens = ["the", "a", "of"]
            res = self.search(health_check_tokens, top_k=1)
            print(f"[OK] Cold start health check completed for {path}. Returned {len(res)} docs.")
        except Exception as he:
            print(f"[Warning] Cold start health check failed for {path}: {he}")


class ShardedBM25:
    def __init__(self, index_dir, shard_size=1000000, k1=1.5, b=0.75):
        self.index_dir = index_dir
        self.shard_size = shard_size
        self.k1 = k1
        self.b = b
        self.shards = []
        self.doc_ids = []
        self.vocab = {}
        
    def index(self, corpus_dataset):
        import math
        num_docs = len(corpus_dataset)
        num_shards = math.ceil(num_docs / self.shard_size)
        print(f"Indexing corpus of size {num_docs} using BM25Index in {num_shards} shards...")
        
        self.shards = []
        self.doc_ids = []
        self.vocab = {}
        
        os.makedirs(self.index_dir, exist_ok=True)
        for s in range(num_shards):
            start_idx = s * self.shard_size
            end_idx = min(start_idx + self.shard_size, num_docs)
            shard_docs = corpus_dataset[start_idx:end_idx]
            
            shard_index = BM25Index(k1=self.k1, b=self.b)
            shard_index.index(shard_docs)
            
            shard_path = os.path.join(self.index_dir, f"shard_{s}")
            shard_index.save(shard_path)
            
            self.shards.append(shard_index)
            self.doc_ids.extend(shard_index.doc_ids)
            self.vocab.update(shard_index.vocab)
            
        gc.collect()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for s, shard in enumerate(self.shards):
            shard_path = os.path.join(path, f"shard_{s}")
            shard.save(shard_path)

    def load(self):
        self.shards = []
        self.doc_ids = []
        self.vocab = {}
        
        # 1. Check if there are sharded directories (e.g. shard_0)
        shard_0_path = os.path.join(self.index_dir, "shard_0")
        if os.path.exists(shard_0_path):
            print(f"Detected sharded index at {self.index_dir}. Loading shards...")
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
        else:
            # Single-index loading case
            bm25s_index_file = os.path.join(self.index_dir, "data.csc.index.npy")
            if os.path.exists(bm25s_index_file):
                shard_index = BM25Index(k1=self.k1, b=self.b)
                shard_index.load(self.index_dir)
                self.shards.append(shard_index)
                print("Loaded single BM25 index.")
            else:
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
        
        for shard in self.shards:
            self.doc_ids.extend(shard.doc_ids)
            self.vocab.update(shard.vocab)

    def search(self, query_tokens, top_k=100, doc_mask=None, filter_names=None):
        if not self.shards:
            return []
            
        if len(self.shards) == 1:
            return self.shards[0].search(query_tokens, top_k=top_k, doc_mask=doc_mask, filter_names=filter_names)
            
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
