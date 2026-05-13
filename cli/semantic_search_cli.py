#!/usr/bin/env python3
from lib.semantic_search import (
    verify_model, 
    embed_text,
    verify_embeddings
)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Very current model")
    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_text_parser = subparsers.add_parser("embed_text", help="Use to embed text.")
    embed_text_parser.add_argument('text', type=str, help="Text to embed.")

    embed_query_parser = subparsers.add_parser("embed_query", help="Search for query.")
    embed_query_parser.add_argument('query', type=str, help="Text to embed.")

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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()