import re
import unicodedata
import nltk
from nltk.stem import PorterStemmer
from rapidfuzz import process, utils

STEMMER = PorterStemmer()

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    # Fallback stopwords
    STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
        'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
    }


def tokenize_text(text, lowercase=True, remove_punctuation=True, remove_stopwords=True, use_lemmatization=False):
    if not text:
        return []
    # Strip Penn Treebank tokenization noise (e.g. -LSB-, -RSB-, -LRB-, -RRB-) before tokenization
    text = text.replace("-LSB-", " ").replace("-RSB-", " ")
    text = text.replace("-LRB-", " ").replace("-RRB-", " ")
    text = text.replace("-LCB-", " ").replace("-RCB-", " ")
    
    text = unicodedata.normalize('NFKC', text)
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS or w in {'no', 'not', 'nor', 'against', 'without'}]
    if use_lemmatization:
        tokens = [STEMMER.stem(w) for w in tokens]
    return tokens

class QueryPreprocessor:
    def __init__(self, config):
        self.config = config.get("preprocessing", {})
        self.spell_config = config.get("spell_correction", {})
        self.stemmer = PorterStemmer()
        self.vocab = set()
        self.vocab_list = []
        
    def set_vocab(self, vocab):
        self.vocab = set(vocab)
        self.vocab_list = list(self.vocab)
        
    def clean_query(self, query, correct_spelling=False):
        if not query:
            return ""
            
        # Normalize Unicode
        query = unicodedata.normalize('NFKC', query)
        
        # Lowercase
        if self.config.get("lowercase", True):
            query = query.lower()
            
        # Remove punctuation & control chars
        if self.config.get("remove_punctuation", True):
            query = re.sub(r'[^\w\s]', ' ', query)
            
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Tokenize
        words = self.tokenize_text(query)
        cleaned = " ".join(words)
        
        if correct_spelling:
            return self.correct_spelling(cleaned)
        return cleaned

    def tokenize_text(self, text, lowercase=True, remove_punctuation=True, remove_stopwords=True, use_lemmatization=False):
        return tokenize_text(text, lowercase, remove_punctuation, remove_stopwords, use_lemmatization)
    def correct_spelling(self, query):
        if not self.vocab_list:
            return query
            
        words = query.split()
        corrected_words = []
        min_sim = self.spell_config.get("min_similarity", 0.85) * 100.0
        
        for word in words:
            # If word is already in vocab, no correction needed
            if word in self.vocab or word.isdigit():
                corrected_words.append(word)
                continue
                
            # Perform spell check
            res = process.extractOne(word, self.vocab_list, processor=utils.default_process)
            if res and res[1] >= min_sim:
                corrected_words.append(res[0])
            else:
                corrected_words.append(word)
                
        return " ".join(corrected_words)

def parse_filter(filter_str):
    from src.metadata_parser import MetadataParser
    parser = MetadataParser()
    ast = parser.parse(filter_str)
    # Return a tuple (field, op, val) if it's a simple FilterExpression to preserve backward compatibility for tests
    from src.query_ast import FilterExpression
    if isinstance(ast, FilterExpression):
        return ast.field, ast.operator, ast.value
    return ast

def evaluate_filter(doc_metadata, parsed_filter):
    if not parsed_filter:
        return True
    if not doc_metadata or not isinstance(doc_metadata, dict):
        return False
        
    from src.query_ast import FilterExpression, LogicalExpression, NotExpression
    
    # Check if we got the tuple-based format (field, op, val) for backward compatibility
    if isinstance(parsed_filter, tuple) and len(parsed_filter) == 3:
        field, op, val = parsed_filter
        if field not in doc_metadata:
            return False
        doc_val = doc_metadata[field]
        if op == "==":
            return doc_val == val
        elif op == "!=":
            return doc_val != val
        elif op == ">=":
            return doc_val >= val
        elif op == "<=":
            return doc_val <= val
        elif op == ">":
            return doc_val > val
        elif op == "<":
            return doc_val < val
        return False
        
    # Evaluate AST Nodes
    if isinstance(parsed_filter, FilterExpression):
        field, op, val = parsed_filter.field, parsed_filter.operator, parsed_filter.value
        if field not in doc_metadata:
            return False
        doc_val = doc_metadata[field]
        if op == "==":
            return doc_val == val
        elif op == "!=":
            return doc_val != val
        elif op == ">=":
            return doc_val >= val
        elif op == "<=":
            return doc_val <= val
        elif op == ">":
            return doc_val > val
        elif op == "<":
            return doc_val < val
            
    elif isinstance(parsed_filter, LogicalExpression):
        left_val = evaluate_filter(doc_metadata, parsed_filter.left)
        right_val = evaluate_filter(doc_metadata, parsed_filter.right)
        if parsed_filter.operator == "AND":
            return left_val and right_val
        elif parsed_filter.operator == "OR":
            return left_val or right_val
            
    elif isinstance(parsed_filter, NotExpression):
        return not evaluate_filter(doc_metadata, parsed_filter.operand)
        
    return False

