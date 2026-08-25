"""Seam S4 / Phase 1 gate: V0 retrieval baseline runner (docs/02 §5, §8).

Indexes a golden-spec corpus into Postgres through the real ingestion path,
then runs every answerable golden case as a dense-retrieval query and scores
Recall@k / MRR@k / nDCG@k against gold source documents (seam S10 metrics).

Numbers land in reports/<timestamp>/results.json. The runner reports exactly
what it measured - provider/model are recorded alongside every number, so
hash-provider smoke runs can never masquerade as real baselines.

CLI: python -m eval.runners.retrieval --dataset <golden.jsonl> --spec <spec.json>
"""

import argparse
import asyncio
import json
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from atlas_core.config import Settings
from atlas_core.corpus import CorpusSpec, generate_corpus
from atlas_core.corpus.generate import Section as GeneratedSection
from atlas_core.db.models import Base, Document, DocumentVersion, Organization
from atlas_core.db.session import make_engine
from atlas_core.providers import EmbeddingProvider, HashEmbeddingProvider
from atlas_core.retrieval import DenseRetriever, RetrievalFilters, ensure_hnsw_index
from sqlalchemy import insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from eval.datasets.validate import load_jsonl
from eval.metrics.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank_at_k

DEFAULT_DATABASE_URL = "postgresql+asyncpg://atlas:atlas@localhost:5432/atlas"

# Corpus doc ids ("acme_policy_refund_policy_v3") map deterministically onto
# Document PK UUIDs so golden-case gold_sources can be scored against rankings.
DOC_ID_NAMESPACE = uuid.UUID("6f6e6b6e-6f77-4c65-64a7-e5209c1d5a17")


def corpus_doc_uuid(doc_id: str) -> uuid.UUID:
    return uuid.uuid5(DOC_ID_NAMESPACE, doc_id)


def resolve_provider(name: str, api_key: str, model_override: str) -> EmbeddingProvider:
    if name == "openai":
        if not api_key.strip():
            raise SystemExit("FAIL: --provider openai requires --api-key")
        model = model_override or "text-embedding-3-small"
        from atlas_core.providers import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(api_key=api_key, model=model)
    if model_override:
        raise SystemExit("FAIL: --model override only applies to --provider openai")
    return HashEmbeddingProvider()


def document_to_markdown(sections: Sequence[GeneratedSection]) -> str:
    """Reconstruct upload-format markdown from manifest sections."""
    parts = []
    for section in sections:
        body = "\n\n".join(p.text for p in section.paragraphs)
        parts.append(f"# {section.heading}\n\n{body}")
    return "\n\n".join(parts)


# === SCORING ===


def _dedupe_by_doc(doc_ids: Sequence[str]) -> list[str]:
    """Contract (COORDINATION.md): dedupe fused rankings before scoring."""
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


async def run_baseline(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    dataset_path: Path,
    limit: int | None,
) -> dict[str, object]:
    cases = [case for case in load_jsonl(dataset_path) if case.answerable]
    if limit is not None:
        cases = cases[:limit]

    retriever = DenseRetriever(engine, provider)
    recalls_5: list[float] = []
    recalls_10: list[float] = []
    rrs_10: list[float] = []
    ndcgs_10: list[float] = []
    latencies_ms: list[float] = []
    per_query: list[dict[str, object]] = []

    for case in cases:
        relevant = [str(corpus_doc_uuid(s.doc_id)) for s in case.gold_sources]
        started = time.perf_counter()
        ranked = await retriever.retrieve(case.question, RetrievalFilters(tenant=case.tenant))
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        latencies_ms.append(elapsed_ms)

        ranking = _dedupe_by_doc([str(r.document_id) for r in ranked.results])[:10]
        grades = {doc_id: 1.0 for doc_id in relevant}
        recall5 = recall_at_k(ranking, relevant, 5)
        recall10 = recall_at_k(ranking, relevant, 10)
        rr10 = reciprocal_rank_at_k(ranking, relevant, 10)
        ndcg10 = ndcg_at_k(ranking, grades, 10)
        recalls_5.append(recall5)
        recalls_10.append(recall10)
        rrs_10.append(rr10)
        ndcgs_10.append(ndcg10)
        per_query.append(
            {
                "case_id": case.id,
                "category": case.category,
                "tenant": case.tenant,
                "recall@5": round(recall5, 4),
                "recall@10": round(recall10, 4),
                "rr@10": round(rr10, 4),
                "ndcg@10": round(ndcg10, 4),
                "latency_ms": elapsed_ms,
                "top_doc": ranking[0] if ranking else None,
            }
        )

    def mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def percentile(fraction: float) -> float:
        if not latencies_ms:
            return 0.0
        ordered = sorted(latencies_ms)
        index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
        return ordered[index]

    return {
        "queries": len(cases),
        "recall@5": mean(recalls_5),
        "recall@10": mean(recalls_10),
        "mrr@10": mean(rrs_10),
        "ndcg@10": mean(ndcgs_10),
        "latency_p50_ms": percentile(0.50),
        "latency_p95_ms": percentile(0.95),
        "per_query": per_query,
    }


async def prepare_schema(engine: AsyncEngine, dimension: int) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await ensure_hnsw_index(engine, dimension)


async def _load_org_ids(engine: AsyncEngine) -> dict[str, str]:
    async with engine.connect() as conn:
        rows = (await conn.execute(select(Organization.id, Organization.name))).all()
    return {name: str(org_id) for org_id, name in rows}


async def ingest_corpus(
    engine: AsyncEngine,
    provider: EmbeddingProvider,
    spec: CorpusSpec,
    chunking_strategy: str,
) -> int:
    """Index every manifest document through the real ingestion path."""
    from atlas_api.services.ingestion import process_document  # deferred: app-layer import

    manifest = generate_corpus(spec)

    async with engine.begin() as conn:
        for tenant_id in sorted({d.tenant_id for d in manifest.documents}):
            await conn.execute(
                pg_insert(Organization)
                .values(name=tenant_id)
                .on_conflict_do_nothing(index_elements=[Organization.name])
            )
    org_ids = await _load_org_ids(engine)

    ingested = 0
    for doc in manifest.documents:
        document_id = corpus_doc_uuid(doc.doc_id)
        version_id = uuid.uuid4()
        content = document_to_markdown(doc.sections)
        async with engine.begin() as conn:
            await conn.execute(
                insert(Document).values(
                    id=document_id,
                    organization_id=uuid.UUID(org_ids[doc.tenant_id]),
                    title=doc.title,
                    doc_type=doc.doc_type,
                )
            )
            await conn.execute(
                insert(DocumentVersion).values(
                    id=version_id,
                    document_id=document_id,
                    version_number=doc.version or 1,
                    content_hash=f"{doc.doc_id}:{manifest.spec_hash}",
                    effective_date=doc.effective_date,
                    status="indexing",
                    source_text=content,
                )
            )
        published = await process_document(
            engine,
            provider,
            upload_id=uuid.uuid4(),
            document_id=document_id,
            version_id=version_id,
            content=content,
            strategy_name=chunking_strategy,
        )
        if not published:
            raise RuntimeError(f"ingestion failed for corpus doc {doc.doc_id}")
        ingested += 1
    return ingested


async def run_all(args: argparse.Namespace) -> dict[str, object]:
    provider = resolve_provider(args.provider, args.api_key, args.model)
    probe = await provider.embed(["dimension probe"])
    dimension = len(probe[0].vector)

    engine = make_engine(Settings(database_url=args.database_url).database_url)
    try:
        await prepare_schema(engine, dimension)
        count = await ingest_corpus(engine, provider, args.spec_path, args.chunking_strategy)
        print(
            f"indexed {count} docs | provider={type(provider).__name__} "
            f"model={provider.model_name} dim={dimension} "
            f"chunking={args.chunking_strategy}"
        )
        return await run_baseline(engine, provider, args.dataset_path, args.limit)
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V0 dense retrieval baseline runner")
    parser.add_argument("--dataset", required=True, help="golden JSONL path")
    parser.add_argument("--spec", required=True, help="corpus spec JSON used to build the dataset")
    parser.add_argument("--provider", choices=["hash", "openai"], default="hash")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="", help="override embedding model (openai only)")
    parser.add_argument("--limit", type=int, default=None, help="cap number of queries")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument("--chunking-strategy", default="paragraph")
    args = parser.parse_args(argv)
    args.dataset_path = Path(args.dataset)
    args.spec_path = CorpusSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8"))

    results = asyncio.run(run_all(args))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir = Path(__file__).resolve().parents[1] / "reports" / stamp
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runner": "v0-dense",
        "provider": type(resolve_provider(args.provider, args.api_key, args.model)).__name__,
        "chunking_strategy": args.chunking_strategy,
        **results,
    }
    (report_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"\nV0 baseline ({args.provider}): "
        f"R@5={results['recall@5']} R@10={results['recall@10']} "
        f"MRR@10={results['mrr@10']} nDCG@10={results['ndcg@10']} "
        f"| p50={results['latency_p50_ms']}ms p95={results['latency_p95_ms']}ms "
        f"over {results['queries']} queries"
    )
    print(f"report written: {report_dir / 'results.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
