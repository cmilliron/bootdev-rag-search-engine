import string
from .search_utils import load_movies, load_stop_words
from nltk.stem import PorterStemmer


def search_handler(query, limit=None):
    movies = load_movies();
    stop_words = load_stop_words()
    # print(stop_words)
    # print("Movies: ", movies['movies'][1])
    results = []
    processed_query = tokenize_text(query, stop_words)
    for movie in movies:
        processed_movie_title = tokenize_text(movie["title"], stop_words)
        if has_matching_token(processed_query, processed_movie_title):
            results.append(movie)
    if limit == None:
        return results
    return results[:limit]


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for token in query_tokens:
        for movie_token in title_tokens:
            if token in movie_token:
                return True
    return False


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

