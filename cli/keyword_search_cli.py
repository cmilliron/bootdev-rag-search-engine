import argparse
import json
from lib.handlers import search_handler, build_handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build an inverted index.")

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
