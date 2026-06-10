import os
import yaml
import gc
from functools import lru_cache
from src.preprocessing import QueryPreprocessor, parse_filter
from src.retrievers.bm25 import ShardedBM25
from src.reranker import CrossEncoderReranker
from src.data_loader import CorpusDBHelper

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
        try:
            self.load_indexes()
            
            doc_mask = None
            if metadata_filter:
                try:
                    parsed_filter = parse_filter(metadata_filter)
                    print(f"Applying metadata filter: {parsed_filter}")
                    doc_mask = self.db_helper.get_all_doc_ids_matching_filter(parsed_filter)
                    print(f"Filter matched {len(doc_mask) if doc_mask else 0} documents.")
                    if doc_mask is not None and len(doc_mask) == 0:
                        return []
                except Exception as filter_err:
                    print(f"⚠️ Filter error: {filter_err}. Proceeding without filter.")
                    doc_mask = None
                    
            if not query or not query.strip():
                return []
                
            corrected_query = self.preprocessor.clean_query(query, correct_spelling=correct_spelling)
            print(f"Raw query: '{query}' -> Preprocessed: '{corrected_query}'")
            
            if not corrected_query or not corrected_query.strip():
                # Try with original query if all words were stopwords
                print("⚠️ Query became empty after preprocessing. Retrying with raw terms.")
                raw_tokens = query.lower().split()
                corrected_query = " ".join(raw_tokens[:5])  # Limit to first 5 terms
                
            bm25_k = self.config.get("bm25", {}).get("top_k", 100)
            bm25_results = self.bm25_retriever.search(corrected_query.split(), top_k=bm25_k, doc_mask=doc_mask)
            
            # Fallback strategy
            if not bm25_results:
                raw_tokens = query.lower().split()[:10]
                print(f"📍 No results with preprocessed query, trying raw tokens: {raw_tokens}...")
                bm25_results = self.bm25_retriever.search(raw_tokens, top_k=bm25_k, doc_mask=doc_mask)
            
            if not bm25_results:
                print("❌ No documents found matching the query.")
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
