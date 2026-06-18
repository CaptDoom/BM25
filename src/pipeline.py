import os
import yaml
import gc
import re
from functools import lru_cache
from src.preprocessing import QueryPreprocessor, parse_filter
from src.retrievers.bm25 import ShardedBM25
from src.reranker import CrossEncoderReranker
from src.data_loader import CorpusDBHelper

def parse_embedded_filters(query_str):
    # Matches patterns like @dataset(fever)
    filters = re.findall(r"@(\w+)\(([^)]+)\)", query_str)
    # Clean query from filters
    cleaned = re.sub(r"@\w+\([^)]+\)", "", query_str).strip()
    return cleaned, filters

def translate_filters_to_bitmap_names(filters):
    names = []
    for field, val in filters:
        field = field.lower().strip()
        val = val.lower().strip()
        
        if field == "dataset":
            # 0=FEVER, 1=Quora, 2=SciDocs, 3=FIQA, 4=CQADupstack, 5=Other
            ds_map = {
                "fever": 0, "quora": 1, "scidocs": 2, "fiqa": 3, "cqadupstack": 4
            }
            ds_id = ds_map.get(val, 5)
            names.append(f"ds_{ds_id}")
        elif field == "has_title":
            flag = 1 if val in ("true", "1", "yes") else 0
            names.append(f"has_title_{flag}")
        elif field == "length":
            if val in ("short", "medium", "long"):
                names.append(f"len_{val}")
        elif field == "has_code":
            flag = 1 if val in ("true", "1", "yes") else 0
            names.append(f"has_code_{flag}")
    return names


class RetrievalPipeline:
    def __init__(self, config_path, index_dir):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.index_dir = index_dir
        self.db_path = os.path.join(index_dir, "corpus.db")
        self.db_helper = CorpusDBHelper(self.db_path)
        
        self.preprocessor = QueryPreprocessor(self.config)
        
        bm25_cfg = self.config.get("bm25", {})
        self.bm25_retriever = ShardedBM25(
            index_dir=os.path.join(index_dir, "bm25"),
            shard_size=self.config.get("sharding", {}).get("shard_size", 1000000),
            k1=bm25_cfg.get("k1", 1.2),
            b=bm25_cfg.get("b", 0.75)
        )
        
        rerank_cfg = self.config.get("reranker", {})
        self.reranker = CrossEncoderReranker(
            model_name=rerank_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            batch_size=rerank_cfg.get("batch_size", 32)
        )
        
        self.loaded = False

    def load_indexes(self):
        if not self.loaded:
            print("Loading pipeline indexes...")
            self.bm25_retriever.load()
            
            vocab = set()
            for shard in self.bm25_retriever.shards:
                vocab.update(shard.vocab.keys())
            self.preprocessor.set_vocab(vocab)
            self.search.cache_clear()  # Clear query cache when loading
            self.loaded = True
            print("Pipeline indexes loaded successfully.")
            
    @lru_cache(maxsize=1024)
    def search(self, query, metadata_filter=None, top_k=10, use_reranker=False, correct_spelling=False):
        import time
        start_time = time.time()
        timeout_limit = 5.0 # Max 5 seconds execution limit per query thread context
        
        try:
            self.load_indexes()
            
            # 1. Parse Metadata Filter into AST
            doc_mask = None
            if metadata_filter:
                try:
                    from src.metadata_parser import MetadataParser
                    parser = MetadataParser()
                    parsed_filter = parser.parse(metadata_filter)
                    print(f"Applying metadata AST filter: {parsed_filter}")
                    doc_mask = self.db_helper.get_all_doc_ids_matching_filter(parsed_filter)
                    print(f"Filter matched {len(doc_mask) if doc_mask else 0} documents.")
                    if doc_mask is not None and len(doc_mask) == 0:
                        return []
                except Exception as filter_err:
                    print(f"⚠️ Filter parsing error: {filter_err}. Proceeding without filter.")
                    doc_mask = None
                    
            if not query or not query.strip():
                return []
                
            # Parse embedded query filters
            query, embedded_filters = parse_embedded_filters(query)
            filter_names = translate_filters_to_bitmap_names(embedded_filters)
            if filter_names:
                print(f"Parsed query-embedded filters: {embedded_filters} -> {filter_names}")
                
            # 2. Query Normalization using QueryTransformer
            from src.query_transformer import QueryTransformer
            p_cfg = self.config.get("preprocessing", {})
            transformer = QueryTransformer(
                lowercase=p_cfg.get("lowercase", True),
                remove_punctuation=p_cfg.get("remove_punctuation", True),
                remove_stopwords=p_cfg.get("remove_stopwords", True)
            )
            corrected_query = transformer.normalize(query)
            if correct_spelling:
                corrected_query = self.preprocessor.correct_spelling(corrected_query)
            print(f"Raw query: '{query}' -> Preprocessed: '{corrected_query}'")
            
            if not corrected_query or not corrected_query.strip():
                print("⚠️ Query became empty after preprocessing. Retrying with raw terms.")
                raw_tokens = query.lower().split()
                corrected_query = " ".join(raw_tokens[:5])
                
            # Check timeout
            if time.time() - start_time > timeout_limit:
                print("⚠️ Search execution timeout limit exceeded during preprocessing.")
                return []
                
            bm25_k = self.config.get("bm25", {}).get("top_k", 100)
            bm25_results = self.bm25_retriever.search(corrected_query.split(), top_k=bm25_k, doc_mask=doc_mask, filter_names=filter_names)
            
            # Fallback strategy
            if not bm25_results:
                raw_tokens = query.lower().split()[:10]
                print(f"📍 No results with preprocessed query, trying raw tokens: {raw_tokens}...")
                bm25_results = self.bm25_retriever.search(raw_tokens, top_k=bm25_k, doc_mask=doc_mask, filter_names=filter_names)
            
            if not bm25_results:
                print("❌ No documents found matching the query.")
                return []
                
            # Check timeout
            if time.time() - start_time > timeout_limit:
                print("⚠️ Search execution timeout limit exceeded during BM25 search.")
                return []
            
            # Reranking with error handling
            reranked_results = None
            if use_reranker:
                try:
                    rerank_depth = self.config.get("reranker", {}).get("top_n", 25)
                    bm25_candidates = bm25_results[:rerank_depth]
                    reranked_results = self.reranker.rerank(
                        query=corrected_query,
                        candidates=bm25_candidates,
                        corpus_db_helper=self.db_helper,
                        top_k=top_k
                    )
                except Exception as rerank_err:
                    print(f"⚠️ Reranking failed: {rerank_err}. Using BM25 results.")
                    reranked_results = bm25_results[:top_k]
            else:
                reranked_results = bm25_results[:top_k]
            
            # Lazy hydration - load document text only for the final top_k documents
            final_doc_ids = [doc_id for doc_id, _ in reranked_results]
            enriched_docs = self.db_helper.get_documents(final_doc_ids)
            
            results = []
            for doc_id, score in reranked_results:
                doc_data = enriched_docs.get(doc_id, {"title": "", "text": "", "metadata": {}})
                results.append({
                    "id": doc_id,
                    "score": score,
                    "title": doc_data.get("title", ""),
                    "text": doc_data.get("text", ""),
                    "metadata": doc_data.get("metadata", {})
                })
                
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

