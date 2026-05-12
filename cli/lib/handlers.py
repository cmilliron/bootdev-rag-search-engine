
import pickle
import math
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
import string
import sys
from .search_utils import (
    load_stop_words, 
    load_movies, 
    format_search_result,
    INDEX_CACHE_PATH, 
    DOCMAP_CACHE_PATH, 
    TF_CACHE_PATH,
    CACHE_DIR,
    BM25_K1,
    BM25_B,
    DOC_LENGTHS_PATH
)

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.doc_map: dict[int, dict] = {}
        self.term_frequency = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_path = INDEX_CACHE_PATH
        self.docmap_path =  DOCMAP_CACHE_PATH
        self.tf_path = TF_CACHE_PATH
        self.doc_lengths_path = DOC_LENGTHS_PATH
        

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
    

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        doc_len = self.doc_lengths.get(doc_id, 0)
        ave_doc_len = self.__get_avg_doc_length()
        if ave_doc_len > 0:
            length_nomralization = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        else:
            length_nomralization = 1
        tf = self.get_tf(doc_id, term)
        bm25_tf = ((tf * (k1 + 1)) / (tf + k1 * length_nomralization))
        return bm25_tf

    
    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
    

    def bm25_search(self, query: str, limit: int = 5):
        tokens = tokenize_text(query)
        bm25_scores: dict[int, float] = {}
        for doc in self.doc_map:
            bm25_total = 0.0
            for token in tokens:
                bm25_score = self.bm25(doc, token)
                bm25_total += bm25_score
            bm25_scores[doc] = bm25_total
        sorted_scores = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        
        for doc_id, score in sorted_scores[:limit]:
            movie = self.doc_map[doc_id]
            formated_result = format_search_result(
                doc_id= doc_id, 
                title= movie["title"], 
                description=movie["description"], 
                score=score
            )
            results.append(formated_result)
        return results


    def __add_documents(self, doc_id, text):
        words = tokenize_text(text)
        self.doc_lengths[doc_id] = len(words)
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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)


    def load(self):
        try:
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, 'rb') as f:
                self.doc_map = pickle.load(f)
            with open(self.tf_path, 'rb') as f:
                self.term_frequency = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileExistsError:
            print("The file was missing! Using default empty data.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred while loading: {e}")  
            sys.exit(1)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_map) == 0:
            return 0.0
        num_docs = len(self.doc_map)
        total_len = 0
        for doc in self.doc_lengths:
            total_len += self.doc_lengths[doc]
        ave_length = total_len / num_docs
        return ave_length


#  Command Handler Functions
def search_handler(query, limit=5):
    idx = InvertedIndex()
    idx.load()
    seen, results = set(), []
    tokenized_query = tokenize_text(query)
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


def bm25_tf_handler(doc_id: int, term:str, k1:float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)


def bm25search_handler(query: str, limit: int= 5):
    idx = InvertedIndex()
    idx.load()
    bm_25_results = idx.bm25_search(query, limit)
    return bm_25_results


# Helper Functions
def prepare_text(query):
    text = query.lower()
    table = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(table)
    return clean_text


def tokenize_text(text: str) -> list[str]:
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