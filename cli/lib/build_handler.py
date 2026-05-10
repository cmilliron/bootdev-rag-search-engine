from .search_handler import tokenize_text
from .search_utils import load_stop_words, load_movies, INDEX_CACHE_PATH, DOCMAP_CACHE_PATH, CACHE_DIR
import pickle
from collections import defaultdict

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.doc_map: dict[int, dict] = {}
        self.index_path = INDEX_CACHE_PATH
        self.docmap_path =  DOCMAP_CACHE_PATH
        

    def get_documents(self, term):
        doc_ids = self.index.get(term.lower(), set())
        return sorted(doc_ids)


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



def build_handler():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    docs = inverted_index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")
    # print(f"Document content 'merida' = {inverted_index.doc_map[docs[0]]}")