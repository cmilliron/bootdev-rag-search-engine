from sentence_transformers import SentenceTransformer
import numpy as np
from lib.search_utils import EMBEDDINGS_CACHE_PATH, load_movies, CACHE_DIR

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = EMBEDDINGS_CACHE_PATH
    

    def generate_embedding(self, text):
        if not text: # or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        embedding = self.model.encode(text, show_progress_bar=True)
        return embedding

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        mov_list = []
        for doc in documents: 
            self.document_map[doc['id']] = doc
            mov_list.append(f"{doc['title']}: {doc['description']}")    
        self.embeddings = self.generate_embedding(mov_list)
        self.save_embedding()
        return self.embeddings
    
    
    def save_embedding(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)


    def load_or_create_embedding(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents: 
            self.document_map[doc['id']] = doc
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)
    


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embedding(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")     


def embed_query_text(query):
    ss = SemanticSearch()
    query_embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {query_embedding[:3]}")
    print(f"Shape: {query_embedding.shape}")