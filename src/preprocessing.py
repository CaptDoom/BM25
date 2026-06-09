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
        
    def clean_query(self, query):
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
        return " ".join(words)

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
    if not filter_str:
        return None
    match = re.match(r"(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)", filter_str)
    if not match:
        raise ValueError(f"Invalid filter syntax: {filter_str}")
    field, op, val = match.groups()
    val = val.strip().strip("'\"")
    try:
        if "." in val:
            val = float(val)
        else:
            val = int(val)
    except ValueError:
        pass
    return field, op, val

def evaluate_filter(doc_metadata, parsed_filter):
    if not parsed_filter:
        return True
    if not doc_metadata or not isinstance(doc_metadata, dict):
        return False
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
