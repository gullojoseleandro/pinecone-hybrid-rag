# Pinecone hybrid retrieval

A Python 3.12 retrieval module that ingests technical documents into a Pinecone Serverless index, fuses dense vector search with local BM25, and evaluates five labeled questions. It returns source documents rather than generating an LLM answer; answer generation was the preceding RAG exercise.

## Quick path

1. Create a Pinecone account/project and an OpenAI API key. Creating an index, embedding documents, and querying may incur charges.
2. Create a Python 3.12 environment and install dependencies.
3. Copy `.env.example` to `.env` and fill in your own keys.
4. Create/verify the index, ingest the sample corpus, then evaluate.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env locally: PINECONE_API_KEY, OPENAI_API_KEY, INDEX_NAME.
python setup_index.py
python ingest.py
python query.py "What is the Redis cache TTL?"
python evaluate.py
```

On macOS/Linux, replace the first three setup commands with `python3.12 -m venv .venv`, `source .venv/bin/activate`, `python -m pip install -r requirements.txt`, and `cp .env.example .env`.

The setup script creates a **1536-dimensional cosine Serverless index** in the configured cloud and region if it does not exist. If it already exists, it validates its dimension, metric, cloud, and region. It never silently reuses an incompatible index. The default namespace is `coderhouse-demo`; change it in `.env` if you share an index with another dataset.

## What each file does

| File | Role |
| --- | --- |
| `setup_index.py` | Creates or validates the Pinecone Serverless index |
| `ingest.py` | Loads `.md`, `.json`, and `.pdf`, chunks at 650 tokens with 75-token overlap, embeds, and upserts |
| `retriever.py` | `RAGSystem`: Pinecone dense retrieval + BM25 via `EnsembleRetriever`, top 5 |
| `evaluate.py` | Precision@k and Recall@k over `golden_set.json` |
| `query.py` | Prints the top five retrieved chunks with source metadata |
| `data/` | Ten small technical Markdown documents |

Each Pinecone vector stores its original chunk text and metadata: `source`, `document_id`, `page` (1-based for PDFs, 0 for Markdown/JSON), `category`, `tags`, `chunk_id`, and a corpus fingerprint. The local `corpus_manifest.json` stores the same chunks and fingerprint for BM25 and is ignored by Git. Keep this file beside your application when querying; run ingestion again to rebuild it. Only the selected namespace and matching corpus fingerprint are queried.

For a custom corpus, use `python ingest.py --data path/to/documents --manifest path/to/corpus_manifest.json`. JSON files contain an object or list of objects with a string `content` field and optional `category`. Empty PDFs or image-only PDFs need OCR before this loader can index them.

## Evaluation

The five questions in `golden_set.json` each identify one relevant source document. For each question, **Recall@5** is 1 when at least one chunk from that document appears in the top five; otherwise 0. **Precision@5** is the number of retrieved chunks from that document divided by 5. The report averages each metric across the five questions. With one relevant chunk per question, the best possible Precision@5 is **0.200**; this is a property of the labeling scheme, not a model ceiling.

The offline test corpus with simulated embeddings and a simulated Pinecone index yields **5/5 hits, Recall@5 = 1.000, Precision@5 = 0.200**. These are **test-fixture results, not cloud measurements**. After you populate your own index, `python evaluate.py` prints the real Pinecone-backed metrics:

```text
HIT cache/redis.md: What is the TTL of the Redis order-summary cache?
...
Precision@5: 0.200
Recall@5: 1.000
Hits: 5/5
```

That block illustrates the output format; actual scores may differ. Pinecone writes may take a short time to become queryable, so wait briefly and rerun evaluation if it immediately misses newly ingested records.

Run all offline tests without keys or cloud calls:

```powershell
python -m unittest discover -s tests -v
```

## Operational notes

- The same `text-embedding-3-small` model is used for indexing and querying. Do not query an existing namespace with vectors from a different embedding model.
- Vector IDs are stable for identical chunks, so repeated upserts overwrite rather than duplicate them. A corpus fingerprint filters out stale vectors after re-ingestion into the same namespace. The script does **not** delete old vectors; they still consume storage. Use a fresh namespace for a changed corpus, or clean old records deliberately in Pinecone.
- BM25 is built from the local manifest. This is practical for a small-to-medium corpus; for a very large distributed corpus, move lexical search to a shared sparse/full-text service instead of copying the entire manifest to every instance.
- `.env` and `corpus_manifest.json` are ignored by Git. Never commit API keys. The sample documents are safe to publish.
- `pinecone>=9,<10` is intentional: version 9's Serverless index plus vector `upsert/query` interface matches this exercise. Version 10 introduces a document-schema creation path that is not interchangeable with vector operations. See the [Pinecone v9 migration guide](https://sdk.pinecone.io/python/migration/v9-migration.html) and [v10 migration guide](https://sdk.pinecone.io/python/migration/v10-migration.html).
