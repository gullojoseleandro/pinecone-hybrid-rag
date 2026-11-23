"""Create or validate a dense Pinecone Serverless index."""

from pinecone import Pinecone, ServerlessSpec

from config import EMBEDDING_DIMENSION, load_settings


class IndexMismatch(ValueError):
    """Existing index cannot store the configured embedding vectors."""


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def ensure_index(client: Pinecone, name: str, cloud: str, region: str) -> None:
    if not client.indexes.exists(name):
        client.indexes.create(
            name=name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
            timeout=120,
        )
    validate_index(client.indexes.describe(name), name, cloud, region)


def validate_index(description, name: str, cloud: str, region: str) -> None:
    serverless = _field(description.spec, "serverless")
    if (
        description.dimension != EMBEDDING_DIMENSION
        or description.metric != "cosine"
        or not serverless
        or _field(serverless, "cloud") != cloud
        or _field(serverless, "region") != region
    ):
        raise IndexMismatch(
            f"Index {name!r} must be Serverless {cloud}/{region}, "
            f"cosine, dimension {EMBEDDING_DIMENSION}."
        )


def main() -> None:
    settings = load_settings()
    client = Pinecone(api_key=settings.pinecone_api_key)
    ensure_index(client, settings.index_name, settings.cloud, settings.region)
    print(f"Index {settings.index_name!r} is ready in {settings.cloud}/{settings.region}.")


if __name__ == "__main__":
    main()
