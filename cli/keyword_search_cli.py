import argparse
import json
from lib.handlers import (
    search_handler, 
    build_handler, 
    tf_search_handler,
    idf_handler,
    tfidf_handler,
    bm25_idf_handler,
    bm25_tf_handler,
    bm25search_handler
    )
from lib.search_utils import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build an inverted index.")

    tf_search_parser = subparsers.add_parser("tf", help="Search for term frequency.")
    tf_search_parser.add_argument("doc_id", type=int, help="Search query")
    tf_search_parser.add_argument("tf_term", type=str, help="Search query")

    idf_search_parser = subparsers.add_parser("idf", help="Shows the inverse document frequence (IDF)")
    idf_search_parser.add_argument("idf_query", type=str, help="Term to calculate IDF")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF score for a given document ID and term.")
    tfidf_parser.add_argument("doc_id", type=int, help="Document Id.")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for.")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")           

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("-l", "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search Limit" )

    args = parser.parse_args()


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_handler(args.query, 5)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")

        case "build":
            print("Building inverted index...")
            build_handler()
            print("Inverted index built successfully and picked.")

        case "tf":
            tf = tf_search_handler(args.doc_id, args.tf_term)
            print(f"Term frequency of '{args.tf_term}' in document '{args.doc_id}': {tf}")
        
        case 'idf':
            idf = idf_handler(args.idf_query);
            print(f"Inverse document frequency of '{args.idf_query}': {idf:.2f}")

        case "tfidf":
            tf_idf = tfidf_handler(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25_idf = bm25_idf_handler(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")

        case "bm25tf":
            bm25_tf = bm25_tf_handler(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")

        case "bm25search":
            results = bm25search_handler(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. ({result["id"]}) {result["title"]} - Score: {result["score"]:.2f}")
        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
