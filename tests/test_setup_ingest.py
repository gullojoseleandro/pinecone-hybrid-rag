import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ingest import build_chunks, ingest_documents, load_documents
from setup_index import IndexMismatch, ensure_index
from tests.helpers import FakeClient, FakeEmbeddings, FakeIndex


class SetupAndIngestTests(unittest.TestCase):
    def test_creates_serverless_1536_index_and_reuses_matching_one(self):
        client = FakeClient()
        ensure_index(client, "demo", "aws", "us-east-1")
        self.assertEqual(len(client.indexes.create_calls), 1)
        self.assertEqual(client.indexes.create_calls[0]["dimension"], 1536)
        self.assertEqual(client.indexes.create_calls[0]["spec"].region, "us-east-1")
        ensure_index(client, "demo", "aws", "us-east-1")
        self.assertEqual(len(client.indexes.create_calls), 1)

    def test_rejects_dimension_mismatch(self):
        existing = SimpleNamespace(dimension=768, metric="cosine", spec=SimpleNamespace(serverless={"cloud": "aws", "region": "us-east-1"}))
        with self.assertRaises(IndexMismatch):
            ensure_index(FakeClient(existing), "demo", "aws", "us-east-1")

    def test_loads_markdown_json_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "api").mkdir()
            (root / "api" / "fastapi.md").write_text("# FastAPI\nDepends injects dependencies.", encoding="utf-8")
            (root / "api" / "notes.json").write_text(json.dumps({"content": "Redis cache", "category": "cache"}), encoding="utf-8")
            docs = load_documents(root)
            chunks = build_chunks(docs)
            self.assertEqual(len(chunks), 2)
            self.assertEqual({doc.metadata["source"] for doc in chunks}, {"api/fastapi.md", "api/notes.json"})
            self.assertEqual({doc.metadata["page"] for doc in chunks}, {0})
            self.assertEqual({doc.metadata["category"] for doc in chunks}, {"api", "cache"})
            self.assertTrue(all(doc.metadata["chunk_id"] for doc in chunks))

    def test_pdf_loader_retains_page_numbers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manual.pdf").write_bytes(b"mocked PDF")
            pages = [
                SimpleNamespace(extract_text=lambda: "First page technical note."),
                SimpleNamespace(extract_text=lambda: "Second page technical note."),
            ]
            with patch("ingest.PdfReader", return_value=SimpleNamespace(pages=pages)):
                docs = load_documents(root)
            self.assertEqual([doc.metadata["page"] for doc in docs], [1, 2])
            self.assertEqual({doc.metadata["source"] for doc in docs}, {"manual.pdf"})

    def test_ingest_stores_text_metadata_namespace_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "redis.md").write_text("Redis caches summaries for 60 seconds.", encoding="utf-8")
            index = FakeIndex()
            manifest = root / "manifest.json"
            count = ingest_documents(data, index, FakeEmbeddings(), "tenant-a", "demo", manifest)
            self.assertEqual(count, 1)
            self.assertEqual(index.last_namespace, "tenant-a")
            item = next(iter(index.vectors.values()))
            self.assertEqual(item["metadata"]["text"], "Redis caches summaries for 60 seconds.")
            self.assertEqual(item["metadata"]["source"], "redis.md")
            self.assertEqual(item["metadata"]["page"], 0)
            self.assertTrue(item["metadata"]["category"])
            self.assertTrue(manifest.exists())
