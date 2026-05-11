
import pickle
import math
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
import string
import sys
from .search_utils import (
    load_stop_words, 
    load_movies, 
    INDEX_CACHE_PATH, 
    DOCMAP_CACHE_PATH, 
    TF_CACHE_PATH,
    CACHE_DIR
)

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.doc_map: dict[int, dict] = {}
        self.term_frequency = defaultdict(Counter)
        self.index_path = INDEX_CACHE_PATH
        self.docmap_path =  DOCMAP_CACHE_PATH
        self.tf_path = TF_CACHE_PATH
        

    def get_documents(self, term):
        doc_ids = self.index.get(term.lower(), set())
        return sorted(list(doc_ids))
    

    def get_tf(self, doc_id, term) -> int:
        # stop_words = load_stop_words()
        tokenized_term = tokenize_text(term)
        if len(tokenized_term) > 1:
            raise Exception("More than one term.")
        search_term = tokenized_term[0]

        return self.term_frequency.get(doc_id)[search_term] # type: ignore
        # return (self.term_frequency.get(doc_id).get(search_term)) # type: ignore


    def get_idf(self, term:str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        num_docs = len(self.doc_map)
        term_doc_count = len(self.index[token])
        idf = math.log((num_docs + 1) / (term_doc_count + 1))
        return idf
        

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        # tokens = tokenize_text(term)
        # tf_idf_total = 0.0    # counter for total
        # for token in tokens:
        #     tf = self.get_tf(doc_id, token)
        #     idf = self.get_idf(token)
        #     tf_idf = tf * idf
        #     tf_idf_total += tf_idf
        # return tf_idf_total 
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf


    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise Exception("Can only have one term.")
        token = tokens[0]
        df = len(self.index[token])
        total_docs = len(self.doc_map)
        bm25_idf = math.log((total_docs - df + .5)/(df +.5) + 1)
        return bm25_idf


    def __add_documents(self, doc_id, text):
        words = tokenize_text(text)
        for word in words: 
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)
            self.term_frequency[doc_id][word] += 1
            # print(self.term_frequency[doc_id])


    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            movie_text = f"{movie['title']} {movie['description']}"
            self.doc_map[doc_id] = movie
            self.__add_documents(doc_id, movie_text)


    def save(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.doc_map, f)
        with open(self.tf_path, 'wb') as f:
            pickle.dump(self.term_frequency, f)


    def load(self):
        try:
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, 'rb') as f:
                self.doc_map = pickle.load(f)
            with open(self.tf_path, 'rb') as f:
                self.term_frequency = pickle.load(f)
        except FileExistsError:
            print("The file was missing! Using default empty data.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred while loading: {e}")  
            sys.exit(1)


#  Command Handler Functions
def search_handler(query, limit=5):
    movies = load_movies();
    stop_words = load_stop_words()
    idx = InvertedIndex()
    idx.load()
    seen, results = set(), []
    tokenized_query = tokenize_text(query, stop_words)
    for token in tokenized_query:
        matching_doc_ids = idx.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.doc_map[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results


def build_handler():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()


def tf_search_handler(doc_id: int, term):
    idx = InvertedIndex()
    idx.load()
    # print(idx.term_frequency.get(doc_id).get(term))
    result = idx.get_tf(doc_id, term)
    return result

def idf_handler(term: str):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    return idf


def tfidf_handler(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tf_idf = idx.get_tf_idf(doc_id, term)
    return tf_idf


def bm25_idf_handler(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)    



# Helper Functions
def prepare_text(query):
    text = query.lower()
    table = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(table)
    return clean_text


def tokenize_text(text: str, stop_words: list[str] = []) -> list[str]:
    stop_words = load_stop_words()
    clean_text = prepare_text(text)
    stemmer = PorterStemmer()
    tokenized_text = [stemmer.stem(t) for t in clean_text.split() if len(t) > 0 and t not in stop_words]
    return tokenized_text


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for token in query_tokens:
        for movie_token in title_tokens:
            if token in movie_token:
                return True
    return False