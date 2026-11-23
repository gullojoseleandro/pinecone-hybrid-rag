import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from evaluate import evaluate
from ingest import ingest_documents
from retriever import RAGSystem
from tests.helpers import FakeEmbeddings, FakeIndex


class RetrievalAndEvaluationTests(unittest.TestCase):
    def test_hybrid_retriever_returns_five_unique_docs_in_namespace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            for name in ("fastapi", "redis", "postgresql", "celery", "prometheus", "kafka", "docker", "jwt", "nginx", "kubernetes"):
                (data / f"{name}.md").write_text(f"{name} handles the {name} workflow.", encoding="utf-8")
            index = FakeIndex()
            manifest = root / "manifest.json"
            ingest_documents(data, index, FakeEmbeddings(), "tenant-a", "demo", manifest)
            system = RAGSystem(index, FakeEmbeddings(), manifest, namespace="tenant-a", index_name="demo")
            docs = system.retrieve("What does Redis handle?")
            self.assertEqual(len(docs), 5)
            self.assertEqual(len({doc.metadata["chunk_id"] for doc in docs}), 5)
            self.assertEqual(docs[0].metadata["document_id"], "redis.md")
            self.assertEqual(index.last_namespace, "tenant-a")

    def test_metrics_use_five_as_precision_denominator(self):
        class StubSystem:
            def retrieve(self, query):
                expected = "a.md" if query == "hit" else "other.md"
                return [Document(page_content="", metadata={"document_id": name}) for name in (expected, "b.md", "c.md", "d.md", "e.md")]

        result = evaluate(
            StubSystem(),
            [
                {"pregunta": "hit", "documento_id_esperado": "a.md"},
                {"pregunta": "miss", "documento_id_esperado": "missing.md"},
            ],
            k=5,
        )
        self.assertAlmostEqual(result.precision_at_k, 0.1)
        self.assertAlmostEqual(result.recall_at_k, 0.5)

    def test_precision_counts_each_useful_chunk(self):
        class StubSystem:
            def retrieve(self, _query):
                return [
                    Document(page_content="", metadata={"document_id": name})
                    for name in ("a.md", "a.md", "b.md", "c.md", "d.md")
                ]

        result = evaluate(
            StubSystem(),
            [{"pregunta": "q", "documento_id_esperado": "a.md"}],
            k=5,
        )
        self.assertAlmostEqual(result.precision_at_k, 0.4)
        self.assertAlmostEqual(result.recall_at_k, 1.0)

    def test_manifest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text(json.dumps({"index_name": "demo", "namespace": "tenant-a", "embedding_model": "text-embedding-3-small", "chunks": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                RAGSystem(FakeIndex(), FakeEmbeddings(), path, namespace="tenant-b", index_name="demo")

    def test_rejects_wrong_query_embedding_dimension(self):
        class ShortEmbeddings(FakeEmbeddings):
            def embed_query(self, text):
                return [1.0] * 768

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "redis.md").write_text("Redis caches summaries.", encoding="utf-8")
            manifest = root / "manifest.json"
            index = FakeIndex()
            ingest_documents(data, index, FakeEmbeddings(), "demo", "test-index", manifest)
            system = RAGSystem(index, ShortEmbeddings(), manifest, namespace="demo", index_name="test-index")
            with self.assertRaises(ValueError):
                system.retrieve("Redis")

    def test_reingestion_excludes_stale_dense_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "redis.md").write_text("Redis caches summaries.", encoding="utf-8")
            (second / "postgresql.md").write_text("PostgreSQL stores orders.", encoding="utf-8")
            index = FakeIndex()
            manifest = root / "manifest.json"
            embeddings = FakeEmbeddings()
            ingest_documents(first, index, embeddings, "demo", "test-index", manifest)
            ingest_documents(second, index, embeddings, "demo", "test-index", manifest)
            system = RAGSystem(index, embeddings, manifest, namespace="demo", index_name="test-index")
            sources = {doc.metadata["source"] for doc in system.retrieve("Redis")}
            self.assertEqual(sources, {"postgresql.md"})

    def test_five_question_golden_set_against_sample_corpus_offline(self):
        repository = Path(__file__).resolve().parents[1]
        golden = json.loads((repository / "golden_set.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            index = FakeIndex()
            ingest_documents(repository / "data", index, FakeEmbeddings(), "demo", "test-index", manifest)
            system = RAGSystem(index, FakeEmbeddings(), manifest, namespace="demo", index_name="test-index")
            result = evaluate(system, golden)
        self.assertEqual(len(result.queries), 5)
        self.assertEqual(result.recall_at_k, 1.0)
        self.assertAlmostEqual(result.precision_at_k, 0.2)
