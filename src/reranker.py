import os
import torch
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        
    def init_model(self):
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading CrossEncoder model {self.model_name} on {device}...")
            
            if device == "cpu":
                try:
                    import psutil
                    cores = psutil.cpu_count(logical=False) or 4
                except ImportError:
                    cores = os.cpu_count() or 4
                torch.set_num_threads(cores)
                print(f"Optimized PyTorch threads for CPU execution (threads={cores})")
                
            self.model = CrossEncoder(self.model_name, device=device)
            
    def rerank(self, query, candidates, corpus_db_helper, top_k=10):
        if not candidates:
            return []
            
        self.init_model()
        
        # Look up document texts for the top candidates only
        candidate_ids = [doc_id for doc_id, _ in candidates]
        docs = corpus_db_helper.get_documents(candidate_ids)
        
        pairs = []
        valid_doc_ids = []
        for doc_id in candidate_ids:
            doc = docs.get(doc_id)
            if doc:
                doc_text = doc.get("title", "") + " " + doc.get("text", "")
                pairs.append((query, doc_text))
                valid_doc_ids.append(doc_id)
                
        if not pairs:
            return []
            
        # Predict scores in batches
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        reranked_results = []
        for doc_id, score in zip(valid_doc_ids, scores):
            reranked_results.append((doc_id, float(score)))
            
        sorted_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
