from sentence_transformers import SentenceTransformer
import numpy as np
from lib.search_utils import (
    EMBEDDINGS_CACHE_PATH, 
    load_movies, 
    CACHE_DIR, 
    DEFAULT_SEARCH_LIMIT, 
    DEFAULT_CHUNK_SIZE,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH
    )
import re
import json

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
        if self.embeddings:
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
    

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")
        query_embed = self.generate_embedding(query)
        simularities = []
        for i, e in enumerate(self.embeddings):
            cos_sim = cosine_similarity(query_embed, e)
            simularities.append((cos_sim, self.documents[i]))
        formated_results = []
        for score, movie in sorted(simularities, key=lambda x: x[0], reverse=True)[:limit]:
            formated_results.append({
                "score": score,
                "title": movie["title"],
                "description": movie['description'] 
            })
        return formated_results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embedding_path = CHUNK_EMBEDDINGS_PATH
        self.chunk_metadata_path = CHUNK_METADATA_PATH


    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents: 
            self.document_map[doc['id']] = doc
        chunks = []
        chunks_metadata = [{}]
        for doc_idx, doc in enumerate(self.documents):
            description = doc.get('description', "")
            if not description.strip():
                continue
            semantic_chunks = create_semantic_chunks(description, 4, 1)
            for i, sem_chunk in enumerate(semantic_chunks):
                chunks.append(sem_chunk)
                chunks_metadata.append({
                    "movie_idx": doc_idx,
                    "chunk_idx": i,
                    "total_chunks": len(semantic_chunks)
                })
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata
        self.save_chunk_embedding(len(chunks))
        return self.chunk_embeddings

    
    def save_chunk_embedding(self, num_chunks):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if self.chunk_embeddings is not None:
            np.save(self.chunk_embedding_path, self.chunk_embeddings)
        if self.chunk_metadata is not None:
            with open(self.chunk_metadata_path, "w") as f:
              json.dump({"chunks": self.chunk_metadata, "total_chunks": len(num_chunks)}, f, indent=2)     


    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents: 
            self.document_map[doc['id']] = doc
        if self.chunk_embedding_path.exists() and self.chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(self.chunk_embedding_path)
            with open(self.chunk_metadata_path, "r") as json_file:
                data = json.load(json_file) 
                self.chunk_metadata = data['chunks']
            # if len(self.chunk_embeddings) == len(self.documents):
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)



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


def cosine_similarity_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embedding(documents=movies)
    results = ss.search(query, limit)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']})")
        print(f"    {result["description"][:25]}...")


def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = 0):
    words = text.split()
    num_words = len(words)
    chunks = []
    i = 0
    while i < num_words:
        current_chunk = words[i: i + chunk_size]
        chunks.append(" ".join(current_chunk))
        if i + chunk_size >= num_words:
            break
        i = i - overlap + chunk_size
    return chunks


def chunk_command(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = 0):
    chunks = fixed_size_chunking(text, size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, c in enumerate(chunks, 1):
        print(f"{i}. {c}")
    return chunks


def create_semantic_chunks(text: str, chunk_size: int = 4, overlap: int = 0) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    num_sent = len(sentences)
    chunks = []
    i = 0
    while i < num_sent:
        current_chunk = sentences[i: i + chunk_size]
        chunks.append(" ".join(current_chunk))
        if i + chunk_size >= num_sent:
            break
        i = i - overlap + chunk_size
    return chunks


def semantic_chunk_command(text: str, chunk_size: int = 4, overlap: int = 0):
    chunks = create_semantic_chunks(text, chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, c in enumerate(chunks, 1):
        print(f"{i}. {c}")
    return chunks

def embed_chunk_command():
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    embeddings = chunked_semantic_search.chunk_embeddings
    print(f"Generated {len(embeddings)} chunked embeddings")



# Helper Functions
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
