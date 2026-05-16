#!/usr/bin/env python3
from lib.semantic_search import (
    verify_model, 
    embed_text,
    verify_embeddings,
    cosine_similarity_search,
    chunk_command, 
    semantic_chunk_command,
    embed_chunk_command
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE

import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Very current model")
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")
    subparsers.add_parser('embed_chunks', help="Make embeddings using chunks.")

    embed_text_parser = subparsers.add_parser("embed_text", help="Use to embed text.")
    embed_text_parser.add_argument('text', type=str, help="Text to embed.")

    embed_query_parser = subparsers.add_parser("embed_query", help="Search for query.")
    embed_query_parser.add_argument('query', type=str, help="Text to embed.")

    search_parser = subparsers.add_parser("search", help="Search for query.")
    search_parser.add_argument('query', type=str, help="Text to embed.")
    search_parser.add_argument('-l', '--limit', type=int, default=DEFAULT_SEARCH_LIMIT, help="Search Limit")
   
    chunk_parser = subparsers.add_parser("chunk", help="Command to chunk text.")
    chunk_parser.add_argument('text', type=str, help="Text to embed.")
    chunk_parser.add_argument('-c', '--chunk-size', dest="chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size")
    chunk_parser.add_argument('-o', '--overlap', type=int, default=0, help="Chunk overlap")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Command to chunk text.")
    semantic_chunk_parser.add_argument('text', type=str, help="Text to embed.")
    semantic_chunk_parser.add_argument('-c', '--max-chunk-size', dest="max_chunk_size", type=int, default=4, help="Chunk size")
    semantic_chunk_parser.add_argument('-o', '--overlap', type=int, default=0, help="Chunk overlap")



    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "verify_embeddings":
            verify_embeddings()

        case "embed_text":
            embed_text(args.text)

        case "embed_query":
            embed_text(args.query)

        case "search":
            cosine_similarity_search(args.query, args.limit)

        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)

        case "embed_chunks":
            embed_chunk_command()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()