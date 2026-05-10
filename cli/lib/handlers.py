
import pickle
from collections import defaultdict
from nltk.stem import PorterStemmer
import string
import sys
from .search_utils import (
    load_stop_words, 
    load_movies, 
    INDEX_CACHE_PATH, 
    DOCMAP_CACHE_PATH, 
    CACHE_DIR
)

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.doc_map: dict[int, dict] = {}
        self.index_path = INDEX_CACHE_PATH
        self.docmap_path =  DOCMAP_CACHE_PATH
        

    def get_documents(self, term):
        doc_ids = self.index.get(term.lower(), set())
        return sorted(list(doc_ids))


    def __add_documents(self, doc_id, text):
        stop_words = load_stop_words()
        words = tokenize_text(text, stop_words)
        for word in words: 
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)


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


    def load(self):
        try:
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, 'rb') as f:
                self.doc_map = pickle.load(f)
        except FileExistsError:
            print("The file was missing! Using default empty data.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred while loading: {e}")  
            sys.exit(1)


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


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for token in query_tokens:
        for movie_token in title_tokens:
            if token in movie_token:
                return True
    return False


def build_handler():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()


def prepare_text(query):
    text = query.lower()
    table = str.maketrans("", "", string.punctuation)
    clean_text = text.translate(table)
    return clean_text


def tokenize_text(text: str, stop_words: list[str]) -> list[str]:
    clean_text = prepare_text(text)
    stemmer = PorterStemmer()
    tokenized_text = [stemmer.stem(t) for t in clean_text.split() if len(t) > 0 and t not in stop_words]
    # tokenized_text = []
    # for word in clean_text.split():
    #     if len(word) > 0 and word not in stop_words:
    #         tokenized_text.append(word)
    return tokenized_text
