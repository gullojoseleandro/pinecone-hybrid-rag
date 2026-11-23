"""Evaluate hybrid retrieval against five labeled questions."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from retriever import RAGSystem


@dataclass(frozen=True)
class QueryResult:
    question: str
    expected: str
    found: bool
    precision_at_k: float
    recall_at_k: float


@dataclass(frozen=True)
class EvaluationResult:
    precision_at_k: float
    recall_at_k: float
    queries: list[QueryResult]


def evaluate(system, golden_set: list[dict], *, k: int = 5) -> EvaluationResult:
    if not golden_set:
        raise ValueError("Golden set must contain at least one question.")
    if k < 1 or k > 5:
        raise ValueError("k must be between 1 and 5.")
    rows = []
    for item in golden_set:
        question = item["pregunta"]
        expected = item["documento_id_esperado"]
        retrieved = system.retrieve(question)[:k]
        useful = sum(doc.metadata["document_id"] == expected for doc in retrieved)
        found = useful > 0
        rows.append(
            QueryResult(
                question=question,
                expected=expected,
                found=found,
                precision_at_k=useful / k,
                recall_at_k=1.0 if found else 0.0,
            )
        )
    return EvaluationResult(
        precision_at_k=sum(row.precision_at_k for row in rows) / len(rows),
        recall_at_k=sum(row.recall_at_k for row in rows) / len(rows),
        queries=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Precision@k and Recall@k.")
    parser.add_argument("--golden", default="golden_set.json", help="Labeled question JSON file")
    parser.add_argument("--manifest", default="corpus_manifest.json", help="Local BM25 corpus snapshot")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved chunks, up to 5")
    args = parser.parse_args()
    golden_set = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    result = evaluate(RAGSystem.from_environment(args.manifest), golden_set, k=args.k)
    for row in result.queries:
        print(f"{'HIT' if row.found else 'MISS'} {row.expected}: {row.question}")
    print(f"Precision@{args.k}: {result.precision_at_k:.3f}")
    print(f"Recall@{args.k}: {result.recall_at_k:.3f}")
    print(f"Hits: {sum(row.found for row in result.queries)}/{len(result.queries)}")


if __name__ == "__main__":
    main()
