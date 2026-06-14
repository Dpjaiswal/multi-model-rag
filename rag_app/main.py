from __future__ import annotations

import argparse
from textwrap import dedent

from rag_app.config import load_config
from rag_app.documents import discover_pdf_paths
from rag_app.copilot import run_financial_copilot_with_mode
from rag_app.vectorstore import index_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid LangGraph RAG over financial PDF documents."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-pdfs", help="List all discovered PDF files.")
    subparsers.add_parser("index", help="Chunk PDFs and index them into hybrid Qdrant storage.")

    query_parser = subparsers.add_parser("query", help="Ask a question over the indexed PDFs.")
    query_parser.add_argument("question", help="Question to ask the RAG system.")
    query_parser.add_argument("--top-k", type=int, default=None, help="Final number of reranked chunks to use.")
    query_parser.add_argument(
        "--mode",
        choices=["auto", "legacy", "copilot"],
        default="auto",
        help="auto keeps old Q&A logic and uses new copilot logic for specialized intents.",
    )

    return parser


def handle_list_pdfs() -> int:
    config = load_config()
    for pdf_path in discover_pdf_paths(config.data_dir):
        print(pdf_path)
    return 0


def handle_index() -> int:
    config = load_config()
    config.validate_indexing()
    chunk_count = index_documents(config)
    print(
        dedent(
            f"""
            Indexed {chunk_count} chunks
            Collection: {config.collection_name}
            Data directory: {config.data_dir}
            Dense embedding model: {config.embedding_model}
            Sparse embedding model: {config.sparse_embedding_model}
            """
        ).strip()
    )
    return 0


def handle_query(question: str, top_k: int | None, mode: str) -> int:
    config = load_config()
    config.validate_querying()
    result = run_financial_copilot_with_mode(config, question, top_k or config.rerank_top_k, mode)
    print(f"Intent: {result['intent']}")
    print(result["answer"])
    if result.get("citations"):
        print("\nSources:")
        for citation in result["citations"]:
            print(citation)
    if result.get("suggestions"):
        print("\nSuggestions:")
        for suggestion in result["suggestions"]:
            print(f"- {suggestion}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-pdfs":
        return handle_list_pdfs()
    if args.command == "index":
        return handle_index()
    if args.command == "query":
        return handle_query(args.question, args.top_k, args.mode)

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
