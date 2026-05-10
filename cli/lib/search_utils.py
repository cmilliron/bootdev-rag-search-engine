import json
import os
from pathlib import Path 

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
CACHE_DIR = Path(PROJECT_ROOT) / "cache"
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOP_WORD_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
INDEX_CACHE_PATH = CACHE_DIR / "index.pkl"
DOCMAP_CACHE_PATH = CACHE_DIR / "docmap.pkl"
TF_CACHE_PATH = CACHE_DIR / "term_frequencies.pkl"


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_words() -> list[str]:
    with open(STOP_WORD_PATH, 'r') as f:
        data = f.read().splitlines()
    return data