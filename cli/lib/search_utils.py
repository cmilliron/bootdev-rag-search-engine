import json
import os
from pathlib import Path 
from typing import Any

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
DEFAULT_CHUNK_SIZE = 200

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CACHE_DIR = Path(PROJECT_ROOT) / "cache"
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOP_WORD_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
INDEX_CACHE_PATH = CACHE_DIR / "index.pkl"
DOCMAP_CACHE_PATH = CACHE_DIR / "docmap.pkl"
TF_CACHE_PATH = CACHE_DIR / "term_frequencies.pkl"
DOC_LENGTHS_PATH = CACHE_DIR / "doc_lengths.pkl"
EMBEDDINGS_CACHE_PATH = CACHE_DIR / "movie_embeddings.npy"


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_words() -> list[str]:
    with open(STOP_WORD_PATH, 'r') as f:
        data = f.read().splitlines()
    return data

def format_search_result(doc_id: int, title: str, description: str, score: float) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        description: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": description,
        "score": round(score, SCORE_PRECISION),
    }