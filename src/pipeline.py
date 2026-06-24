import os
import re
import time
import yaml
from functools import lru_cache

from src.data_loader import CorpusDBHelper
from src.metadata_parser import MetadataParser
from src.preprocessing import QueryPreprocessor
from src.query_transformer import QueryTransformer
from src.reranker import CrossEncoderReranker
from src.retrievers.bm25 import ShardedBM25


def parse_embedded_filters(query_str):
    # Matches patterns like @dataset(fever)
    filters = re.findall(r"@(\w+)\(([^)]+)\)", query_str)
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
                "fever": 0,
                "quora": 1,
                "scidocs": 2,
                "fiqa": 3,
                "cqadupstack": 4,
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
        self.metadata_parser = MetadataParser()
        self._metadata_filter_cache = {}

        self.preprocessor = QueryPreprocessor(self.config)

        preprocessing_cfg = self.config.get("preprocessing", {})
        self.query_transformer = QueryTransformer(
            lowercase=preprocessing_cfg.get("lowercase", True),
            remove_punctuation=preprocessing_cfg.get("remove_punctuation", True),
            remove_stopwords=preprocessing_cfg.get("remove_stopwords", True),
        )

        bm25_cfg = self.config.get("bm25", {})
        self.bm25_retriever = ShardedBM25(
            index_dir=os.path.join(index_dir, "bm25"),
            shard_size=self.config.get("sharding", {}).get("shard_size", 1000000),
            k1=bm25_cfg.get("k1", 1.2),
            b=bm25_cfg.get("b", 0.75),
        )

        rerank_cfg = self.config.get("reranker", {})
        self.reranker = CrossEncoderReranker(
            model_name=rerank_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            batch_size=rerank_cfg.get("batch_size", 32),
        )

        self.loaded = False

    def load_indexes(self):
        if self.loaded:
            return

        print("Loading pipeline indexes...")
        self.bm25_retriever.load()

        vocab = set()
        for shard in self.bm25_retriever.shards:
            vocab.update(shard.vocab.keys())
        self.preprocessor.set_vocab(vocab)

        self._metadata_filter_cache.clear()
        self.search.cache_clear()
        self.loaded = True
        print("Pipeline indexes loaded successfully.")

    def _resolve_metadata_filter(self, metadata_filter):
        if not metadata_filter:
            return None

        cached = self._metadata_filter_cache.get(("expr", metadata_filter))
        if cached is not None:
            return cached

        parsed_filter = self.metadata_parser.parse(metadata_filter)
        print(f"Applying metadata AST filter: {parsed_filter}")
        doc_indices = self.db_helper.get_all_doc_indices_matching_filter(parsed_filter)
        print(f"Filter matched {len(doc_indices) if doc_indices else 0} documents.")

        if doc_indices is not None and len(doc_indices) == 0:
            self._metadata_filter_cache[("expr", metadata_filter)] = set()
            return set()

        self._metadata_filter_cache[("expr", metadata_filter)] = doc_indices
        return doc_indices

    def _resolve_structured_metadata_filters(self, structured_filters):
        if not structured_filters:
            return None

        cache_key = ("structured", structured_filters)
        cached = self._metadata_filter_cache.get(cache_key)
        if cached is not None:
            return cached

        filter_map = dict(structured_filters)
        doc_indices = self.db_helper.get_all_doc_indices_for_structured_filters(
            year_range=filter_map.get("year_range"),
            categories=filter_map.get("categories"),
            has_title=filter_map.get("has_title"),
            has_code=filter_map.get("has_code"),
        )
        print(f"Applying structured metadata filters: {filter_map}")
        print(f"Structured filter matched {len(doc_indices) if doc_indices else 0} documents.")

        if doc_indices is not None and len(doc_indices) == 0:
            self._metadata_filter_cache[cache_key] = set()
            return set()

        self._metadata_filter_cache[cache_key] = doc_indices
        return doc_indices

    @lru_cache(maxsize=1024)
    def search(
        self,
        query,
        metadata_filter=None,
        structured_filters=None,
        top_k=10,
        use_reranker=False,
        correct_spelling=False,
    ):
        start_time = time.time()
        timeout_limit = 5.0

        try:
            self.load_indexes()

            doc_mask = None
            if structured_filters:
                try:
                    doc_mask = self._resolve_structured_metadata_filters(structured_filters)
                    if doc_mask is not None and len(doc_mask) == 0:
                        return []
                except Exception as filter_err:
                    print(f"Warning: structured metadata filter error: {filter_err}. Proceeding without filter.")
                    doc_mask = None

            if metadata_filter:
                try:
                    expr_mask = self._resolve_metadata_filter(metadata_filter)
                    if expr_mask is not None:
                        doc_mask = expr_mask if doc_mask is None else doc_mask & expr_mask
                    if doc_mask is not None and len(doc_mask) == 0:
                        return []
                except Exception as filter_err:
                    print(f"Warning: metadata filter error: {filter_err}. Proceeding without filter.")
                    doc_mask = None

            if not query or not query.strip():
                return []

            query, embedded_filters = parse_embedded_filters(query)
            filter_names = translate_filters_to_bitmap_names(embedded_filters)
            if filter_names:
                print(f"Parsed query-embedded filters: {embedded_filters} -> {filter_names}")

            corrected_query = self.query_transformer.normalize(query)
            if correct_spelling:
                corrected_query = self.preprocessor.correct_spelling(corrected_query)
            print(f"Raw query: '{query}' -> Preprocessed: '{corrected_query}'")

            if not corrected_query or not corrected_query.strip():
                print("Warning: query became empty after preprocessing. Retrying with raw terms.")
                raw_tokens = query.lower().split()
                corrected_query = " ".join(raw_tokens[:5])

            if time.time() - start_time > timeout_limit:
                print("Warning: search execution timeout limit exceeded during preprocessing.")
                return []

            bm25_k = self.config.get("bm25", {}).get("top_k", 100)
            bm25_results = self.bm25_retriever.search(
                corrected_query.split(),
                top_k=bm25_k,
                doc_mask=doc_mask,
                filter_names=filter_names,
            )

            if not bm25_results:
                raw_tokens = query.lower().split()[:10]
                print(f"Info: no results with preprocessed query, trying raw tokens: {raw_tokens}...")
                bm25_results = self.bm25_retriever.search(
                    raw_tokens,
                    top_k=bm25_k,
                    doc_mask=doc_mask,
                    filter_names=filter_names,
                )

            if not bm25_results:
                print("No documents found matching the query.")
                return []

            if time.time() - start_time > timeout_limit:
                print("Warning: search execution timeout limit exceeded during BM25 search.")
                return []

            if use_reranker:
                try:
                    rerank_depth = self.config.get("reranker", {}).get("top_n", 25)
                    bm25_candidates = bm25_results[:rerank_depth]
                    reranked_results = self.reranker.rerank(
                        query=corrected_query,
                        candidates=bm25_candidates,
                        corpus_db_helper=self.db_helper,
                        top_k=top_k,
                    )
                except Exception as rerank_err:
                    print(f"Warning: reranking failed: {rerank_err}. Using BM25 results.")
                    reranked_results = bm25_results[:top_k]
            else:
                reranked_results = bm25_results[:top_k]

            final_doc_ids = [doc_id for doc_id, _ in reranked_results]
            enriched_docs = self.db_helper.get_documents(final_doc_ids)

            results = []
            for doc_id, score in reranked_results:
                doc_data = enriched_docs.get(doc_id, {"title": "", "text": "", "metadata": {}})
                results.append(
                    {
                        "id": doc_id,
                        "score": score,
                        "title": doc_data.get("title", ""),
                        "text": doc_data.get("text", ""),
                        "metadata": doc_data.get("metadata", {}),
                    }
                )

            return results

        except Exception as e:
            print(f"Search error: {e}")
            import traceback

            traceback.print_exc()
            return []
