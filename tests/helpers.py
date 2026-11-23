from types import SimpleNamespace

from langchain_core.embeddings import Embeddings

KEYWORDS = ["fastapi", "redis", "postgresql", "celery", "prometheus", "kafka", "docker", "jwt", "nginx", "kubernetes"]


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        words = text.casefold()
        return [float(word in words) for word in KEYWORDS] + [0.0] * (1536 - len(KEYWORDS))


class FakeIndex:
    def __init__(self):
        self.vectors = {}
        self.last_namespace = None

    def upsert(self, *, vectors, namespace):
        self.last_namespace = namespace
        for item in vectors:
            self.vectors[(namespace, item["id"])] = item
        return SimpleNamespace(upserted_count=len(vectors))

    def query(self, *, vector, top_k, namespace, include_metadata, filter=None):
        self.last_namespace = namespace
        scored = [
            SimpleNamespace(id=item["id"], score=sum(a * b for a, b in zip(vector, item["values"])), metadata=item["metadata"])
            for (stored_namespace, _), item in self.vectors.items()
            if stored_namespace == namespace
            and (filter is None or item["metadata"].get("corpus_id") == filter["corpus_id"]["$eq"])
        ]
        scored.sort(key=lambda match: (-match.score, match.id))
        return SimpleNamespace(matches=scored[:top_k])


class FakeIndexes:
    def __init__(self, existing=None):
        self.existing = existing
        self.create_calls = []

    def exists(self, name):
        return self.existing is not None

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        self.existing = SimpleNamespace(dimension=kwargs["dimension"], metric=kwargs["metric"], spec=SimpleNamespace(serverless=kwargs["spec"].asdict()["serverless"]))

    def describe(self, name):
        return self.existing


class FakeClient:
    def __init__(self, existing=None):
        self.indexes = FakeIndexes(existing)
