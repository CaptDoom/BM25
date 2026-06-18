import re
import unicodedata
import nltk
from typing import Set, List
from src.preprocessing import tokenize_text

class QueryTransformer:
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True, remove_stopwords: bool = True, min_similarity: float = 0.85):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.vocab = set()

    def set_vocab(self, vocab: Set[str]):
        self.vocab = vocab

    def normalize(self, query: str) -> str:
        if not query:
            return ""
        # Strip Penn Treebank tokenization noise & escape dialect-specific syntax / wildcards
        query = query.replace("-LSB-", " ").replace("-RSB-", " ")
        query = query.replace("-LRB-", " ").replace("-RRB-", " ")
        query = query.replace("-LCB-", " ").replace("-RCB-", " ")
        
        # Escape wildcards or special query chars
        query = re.sub(r'[*?+^~:\\/()\[\]{}]', ' ', query)
        
        query = unicodedata.normalize('NFKC', query)
        if self.lowercase:
            query = query.lower()
        if self.remove_punctuation:
            query = re.sub(r'[^\w\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        tokens = tokenize_text(query, lowercase=self.lowercase, remove_punctuation=self.remove_punctuation, remove_stopwords=self.remove_stopwords)
        return " ".join(tokens)
