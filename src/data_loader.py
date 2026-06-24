import os
import sqlite3
import json
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def load_corpus(self):
        print(f"Loading corpus for {self.dataset_name}...")
        if self.dataset_name == "BeIR/cqadupstack":
            return load_dataset(self.dataset_name, "english", split="corpus")
        elif self.dataset_name.startswith("BeIR/"):
            return load_dataset(self.dataset_name, "corpus", split="corpus")
        else:
            return load_dataset(self.dataset_name, split="train")

    def load_queries(self):
        print(f"Loading queries for {self.dataset_name}...")
        if self.dataset_name == "BeIR/cqadupstack":
            return load_dataset(self.dataset_name, "english", split="queries")
        elif self.dataset_name.startswith("BeIR/"):
            return load_dataset(self.dataset_name, "queries", split="queries")
        else:
            return load_dataset(self.dataset_name, split="queries")

    def load_qrels(self):
        print(f"Loading qrels for {self.dataset_name}...")
        if self.dataset_name == "BeIR/cqadupstack":
            qrels_dataset_name = f"{self.dataset_name}-qrels"
            for split in ["test", "validation", "dev", "train"]:
                try:
                    ds = load_dataset(qrels_dataset_name, split=split)
                    print(f"Loaded qrels split '{split}' from {qrels_dataset_name}")
                    return ds
                except Exception:
                    continue
            raise ValueError(f"Could not find qrels split for {qrels_dataset_name}")
        elif self.dataset_name.startswith("BeIR/"):
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
        # Auto-create tables/indexes and migrate old databases on initialization
        if os.path.exists(self.db_path):
            try:
                self.create_indexes_if_missing()
            except Exception as e:
                print(f"⚠️ Error creating indexes on load: {e}")
        
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_index INTEGER,
                title TEXT,
                text TEXT,
                metadata TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_index ON documents (doc_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_year ON documents(json_extract(metadata, '$.year'))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_category ON documents(json_extract(metadata, '$.category'))")
        conn.commit()
        conn.close()
        
    def create_indexes_if_missing(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if doc_index column exists
        cursor.execute("PRAGMA table_info(documents)")
        cols = [info[1] for info in cursor.fetchall()]
        if "doc_index" not in cols:
            print("Migrating SQLite schema: adding doc_index column...")
            cursor.execute("ALTER TABLE documents ADD COLUMN doc_index INTEGER")
            cursor.execute("UPDATE documents SET doc_index = rowid - 1")
            
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_index ON documents (doc_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_year ON documents(json_extract(metadata, '$.year'))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_category ON documents(json_extract(metadata, '$.category'))")
        conn.commit()
        conn.close()
        
    def insert_documents(self, documents, start_idx=0):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        data = []
        for idx, doc in enumerate(documents):
            meta_str = json.dumps(doc.get("metadata", {})) if "metadata" in doc else "{}"
            data.append((doc["_id"], start_idx + idx, doc.get("title", ""), doc.get("text", ""), meta_str))
            
        cursor.executemany("""
            INSERT OR REPLACE INTO documents (doc_id, doc_index, title, text, metadata)
            VALUES (?, ?, ?, ?, ?)
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

    def _build_sql_where(self, node):
        from src.query_ast import FilterExpression, LogicalExpression, NotExpression

        sql_ops = {"==": "=", "!=": "!=", ">=": ">=", "<=": "<=", ">": ">", "<": "<"}

        # Helper to construct SQL where clause and parameters recursively.
        if isinstance(node, tuple) and len(node) == 3:
            field, op, value = node
            if op not in sql_ops:
                return "1=0", []
            return f"json_extract(metadata, '$.{field}') {sql_ops[op]} ?", [value]

        if isinstance(node, FilterExpression):
            if node.operator not in sql_ops:
                return "1=0", []
            return f"json_extract(metadata, '$.{node.field}') {sql_ops[node.operator]} ?", [node.value]

        if isinstance(node, LogicalExpression):
            left_clause, left_params = self._build_sql_where(node.left)
            right_clause, right_params = self._build_sql_where(node.right)
            return f"({left_clause} {node.operator} {right_clause})", left_params + right_params

        if isinstance(node, NotExpression):
            clause, params = self._build_sql_where(node.operand)
            return f"(NOT ({clause}))", params

        return "1=1", []

    def get_all_doc_ids_matching_filter(self, parsed_filter):
        if not parsed_filter:
            return None

        where_clause, params = self._build_sql_where(parsed_filter)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            query = f"SELECT doc_id FROM documents WHERE {where_clause}"
            cursor.execute(query, params)
            matching_ids = {row[0] for row in cursor.fetchall() if row[0] is not None}
        except sqlite3.OperationalError:
            matching_ids = set()
            
        conn.close()
        return matching_ids

    def get_all_doc_indices_matching_filter(self, parsed_filter):
        if not parsed_filter:
            return None

        where_clause, params = self._build_sql_where(parsed_filter)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = f"SELECT doc_index FROM documents WHERE {where_clause}"
            cursor.execute(query, params)
            matching_indices = {
                int(row[0]) for row in cursor.fetchall()
                if row[0] is not None
            }
        except sqlite3.OperationalError:
            matching_indices = set()

        conn.close()
        return matching_indices

    def get_all_doc_indices_for_structured_filters(
        self,
        year_range=None,
        categories=None,
        has_title=None,
        has_code=None,
    ):
        clauses = []
        params = []

        if year_range and len(year_range) == 2:
            start_year, end_year = year_range
            if start_year is not None:
                clauses.append("CAST(json_extract(metadata, '$.year') AS INTEGER) >= ?")
                params.append(int(start_year))
            if end_year is not None:
                clauses.append("CAST(json_extract(metadata, '$.year') AS INTEGER) <= ?")
                params.append(int(end_year))

        if categories:
            category_values = [str(category).strip() for category in categories if str(category).strip()]
            if category_values:
                placeholders = ",".join(["?"] * len(category_values))
                clauses.append(f"LOWER(COALESCE(json_extract(metadata, '$.category'), '')) IN ({placeholders})")
                params.extend([category.lower() for category in category_values])

        if has_title is not None:
            clauses.append(
                "(CASE WHEN COALESCE(TRIM(title), '') = '' THEN 0 ELSE 1 END) = ?"
            )
            params.append(1 if has_title else 0)

        if has_code is not None:
            code_expr = (
                "instr(LOWER(COALESCE(title, '') || ' ' || COALESCE(text, '')), 'std::') > 0 "
                "OR instr(COALESCE(title, '') || ' ' || COALESCE(text, ''), '`') > 0 "
                "OR COALESCE(json_extract(metadata, '$.has_code'), 0) = 1"
            )
            if has_code:
                clauses.append(f"({code_expr})")
            else:
                clauses.append(f"(NOT ({code_expr}))")

        if not clauses:
            return None

        where_clause = " AND ".join(clauses)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = f"SELECT doc_index FROM documents WHERE {where_clause}"
            cursor.execute(query, params)
            matching_indices = {
                int(row[0]) for row in cursor.fetchall()
                if row[0] is not None
            }
        except sqlite3.OperationalError:
            matching_indices = set()

        conn.close()
        return matching_indices


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
