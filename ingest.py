"""Load technical files, chunk them, embed, and upsert to Pinecone."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pypdf import PdfReader

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, load_settings
from setup_index import ensure_index

CHUNK_TOKENS = 650
OVERLAP_TOKENS = 75
BATCH_SIZE = 50


def load_documents(data_dir: str | Path) -> list[Document]:
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Data directory not found: {root}")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".pdf"}
    )
    if not files:
        raise ValueError(f"No .md, .json, or .pdf documents found in {root}")
    documents: list[Document] = []
    for path in files:
        source = path.relative_to(root).as_posix()
        category = source.split("/")[0] if "/" in source else "technical"
        base_metadata = {
            "source": source,
            "document_id": source,
            "category": category,
            "tags": [category, path.suffix.lower().lstrip(".")],
        }
        if path.suffix.lower() == ".pdf":
            for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(Document(page_content=text, metadata={**base_metadata, "page": page_number}))
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("content"), str):
                    raise ValueError(f"JSON document {source} must contain string 'content'.")
                text = item["content"]
                if text.strip():
                    item_category = str(item.get("category") or category)
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                **base_metadata,
                                "category": item_category,
                                "tags": [item_category, "json"],
                                "page": 0,
                            },
                        )
                    )
        else:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                documents.append(Document(page_content=text, metadata={**base_metadata, "page": 0}))
    if not documents:
        raise ValueError("All supported documents are empty.")
    return documents


def build_chunks(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=OVERLAP_TOKENS,
    )
    chunks = splitter.split_documents(documents)
    for position, chunk in enumerate(chunks):
        identifier = hashlib.sha256(
            f"{chunk.metadata['source']}\0{chunk.metadata['page']}\0{position}\0{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        chunk.metadata["chunk_id"] = identifier
    return chunks


def ingest_documents(
    data_dir: str | Path,
    index,
    embeddings: Embeddings,
    namespace: str,
    index_name: str,
    manifest_path: str | Path = "corpus_manifest.json",
) -> int:
    if not namespace.strip():
        raise ValueError("A non-empty namespace is required.")
    chunks = build_chunks(load_documents(data_dir))
    corpus_id = hashlib.sha256(
        "\n".join(sorted(chunk.metadata["chunk_id"] for chunk in chunks)).encode("utf-8")
    ).hexdigest()
    for offset in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[offset : offset + BATCH_SIZE]
        vectors = embeddings.embed_documents([chunk.page_content for chunk in batch])
        if len(vectors) != len(batch) or any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
            raise ValueError(f"Embedding model must produce {EMBEDDING_DIMENSION} values per document.")
        records = [
            {
                "id": chunk.metadata["chunk_id"],
                "values": vector,
                "metadata": {
                    **chunk.metadata,
                    "text": chunk.page_content,
                    "corpus_id": corpus_id,
                },
            }
            for chunk, vector in zip(batch, vectors)
        ]
        response = index.upsert(vectors=records, namespace=namespace)
        if response.upserted_count != len(records):
            raise RuntimeError("Pinecone did not confirm every vector in the batch.")

    manifest = {
        "index_name": index_name,
        "namespace": namespace,
        "embedding_model": EMBEDDING_MODEL,
        "corpus_id": corpus_id,
        "chunks": [
            {"text": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ],
    }
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed documents and upsert them to Pinecone.")
    parser.add_argument("--data", default="data", help="Directory containing .md, .json, and .pdf files")
    parser.add_argument("--manifest", default="corpus_manifest.json", help="Local BM25 corpus snapshot")
    args = parser.parse_args()
    settings = load_settings()
    client = Pinecone(api_key=settings.pinecone_api_key)
    ensure_index(client, settings.index_name, settings.cloud, settings.region)
    index = client.index(settings.index_name)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=settings.openai_api_key)
    count = ingest_documents(
        args.data,
        index,
        embeddings,
        settings.namespace,
        settings.index_name,
        args.manifest,
    )
    print(f"Upserted {count} chunks to index {settings.index_name!r}, namespace {settings.namespace!r}.")


if __name__ == "__main__":
    main()
