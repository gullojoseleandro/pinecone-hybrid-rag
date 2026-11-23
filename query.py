"""Inspect top-five hybrid search results."""

import argparse

from retriever import RAGSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the hybrid Pinecone/BM25 corpus.")
    parser.add_argument("question", help="Technical question")
    parser.add_argument("--manifest", default="corpus_manifest.json", help="Local BM25 corpus snapshot")
    args = parser.parse_args()
    documents = RAGSystem.from_environment(args.manifest).retrieve(args.question)
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata
        print(f"{rank}. {metadata['source']} (page {metadata['page']}, category {metadata['category']})")
        print(document.page_content[:220].replace("\n", " ") + ("..." if len(document.page_content) > 220 else ""))


if __name__ == "__main__":
    main()
