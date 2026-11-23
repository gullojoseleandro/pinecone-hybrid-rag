"""Hybrid Pinecone + local BM25 retrieval using rank fusion."""

import json
import re
from pathlib import Path

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, load_settings
from setup_index import validate_index

TOP_K = 5


def technical_tokens(text: str) -> list[str]:
    return re.findall(r"[\w./-]+", text.casefold())


class RAGSystem:
    """Fuse semantic Pinecone results with BM25 over the ingested corpus snapshot."""

    def __init__(
        self,
        index,
        embeddings: Embeddings,
        manifest_path: str | Path,
        *,
        namespace: str,
        index_name: str,
    ) -> None:
        manifest_file = Path(manifest_path)
        if not manifest_file.is_file():
            raise ValueError("Local BM25 manifest missing. Run 'python ingest.py' first.")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (
            manifest.get("namespace") != namespace
            or manifest.get("index_name") != index_name
            or manifest.get("embedding_model") != EMBEDDING_MODEL
            or not manifest.get("corpus_id")
        ):
            raise ValueError("Manifest index, namespace, or embedding model does not match this query.")
        chunks = manifest.get("chunks", [])
        if not chunks:
            raise ValueError("The BM25 manifest has no chunks.")
        documents = [
            Document(page_content=item["text"], metadata=item["metadata"])
            for item in chunks
        ]
        self.index = index
        self.embeddings = embeddings
        self.namespace = namespace
        self.corpus_id = manifest["corpus_id"]
        lexical = BM25Retriever.from_documents(
            documents, preprocess_func=technical_tokens, k=TOP_K
        )
        dense = RunnableLambda(self._dense_retrieve)
        self.ensemble = EnsembleRetriever(
            retrievers=[dense, lexical],
            weights=[0.5, 0.5],
            id_key="chunk_id",
        )

    def _dense_retrieve(self, query: str) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(f"Query embedding must have {EMBEDDING_DIMENSION} dimensions.")
        response = self.index.query(
            vector=vector,
            top_k=TOP_K,
            namespace=self.namespace,
            include_metadata=True,
            filter={"corpus_id": {"$eq": self.corpus_id}},
        )
        documents = []
        for match in response.matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop("text", None)
            if not isinstance(text, str):
                raise ValueError(f"Pinecone match {match.id!r} is missing text metadata.")
            metadata.setdefault("chunk_id", match.id)
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def retrieve(self, query: str) -> list[Document]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        return self.ensemble.invoke(query)[:TOP_K]

    @classmethod
    def from_environment(cls, manifest_path: str | Path = "corpus_manifest.json") -> "RAGSystem":
        settings = load_settings()
        client = Pinecone(api_key=settings.pinecone_api_key)
        if not client.indexes.exists(settings.index_name):
            raise ValueError("Pinecone index missing. Run 'python setup_index.py' first.")
        validate_index(
            client.indexes.describe(settings.index_name),
            settings.index_name,
            settings.cloud,
            settings.region,
        )
        index = client.index(settings.index_name)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=settings.openai_api_key)
        return cls(
            index,
            embeddings,
            manifest_path,
            namespace=settings.namespace,
            index_name=settings.index_name,
        )
