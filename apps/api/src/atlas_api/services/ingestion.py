"""Ingestion processing: parse -> chunk (v0 stub) -> embed -> atomic publish."""

import hashlib
import logging
import uuid

from atlas_core.db.models import Chunk, Document, DocumentVersion, Upload
from atlas_core.providers import EmbeddingProvider
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)


def split_sections(content: str) -> list[tuple[str, list[str], int]]:
    """Minimal structure-aware parse: markdown ATX headings delimit sections.

    Returns (heading, paragraphs, page) tuples with pages assigned per section
    ordinal. Content without headings lands in a single 'Content' section.
    Replaced/deepened by the full S3 chunker.
    """
    sections: list[tuple[str, list[str], int]] = []
    current_heading = "Content"
    current_paragraphs: list[str] = []

    def flush() -> None:
        if current_paragraphs:
            sections.append(
                (current_heading, list(current_paragraphs), max(1, len(sections) + 1))
            )

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            flush()
            current_heading = stripped.lstrip("#").strip() or "Content"
            current_paragraphs = []
        elif stripped:
            current_paragraphs.append(stripped)
    flush()

    return sections


def approximate_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def _mark_failure(
    engine: AsyncEngine, version_id: uuid.UUID, upload_id: uuid.UUID, detail: str
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            update(DocumentVersion)
            .where(DocumentVersion.id == version_id)
            .values(status="failed")
        )
        await conn.execute(
            update(Upload)
            .where(Upload.id == upload_id)
            .values(status="failed", error_detail=detail[:2000])
        )


async def process_document(
    engine: AsyncEngine,
    embedding_provider: EmbeddingProvider,
    *,
    upload_id: uuid.UUID,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    content: str,
) -> bool:
    """Process one version. Returns True when published.

    Atomic publication contract: chunk rows and the status flip to 'published'
    happen inside a single transaction; any failure leaves the version
    unpublished and invisible to search.
    """
    try:
        sections = split_sections(content)
        if not sections:
            raise ValueError("document content is empty after parsing")

        chunk_rows: list[dict[str, object]] = []
        global_index = 0
        for heading, paragraphs, page in sections:
            embeddings = await embedding_provider.embed(paragraphs)
            for paragraph, embedding in zip(paragraphs, embeddings, strict=True):
                chunk_rows.append(
                    {
                        "document_version_id": version_id,
                        "chunk_index": global_index,
                        "text": paragraph,
                        "token_count": approximate_token_count(paragraph),
                        "page_number": page,
                        "section_path": [heading],
                        "metadata_json": {
                            "document_id": str(document_id),
                            "version_id": str(version_id),
                            "page": page,
                            "embedding_model": embedding.model,
                        },
                    }
                )
                global_index += 1

        async with engine.begin() as conn:
            if chunk_rows:
                await conn.execute(insert(Chunk), chunk_rows)
            await conn.execute(
                update(DocumentVersion)
                .where(DocumentVersion.id == version_id)
                .values(status="published")
            )
            await conn.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(current_version_id=version_id, status="active")
            )
            await conn.execute(
                update(Upload).where(Upload.id == upload_id).values(status="completed")
            )
        return True
    except Exception as exc:
        logger.exception("ingestion failed for version %s", version_id)
        await _mark_failure(engine, version_id, upload_id, repr(exc))
        return False


async def load_published_chunk_texts(
    engine: AsyncEngine, version_id: uuid.UUID
) -> list[str]:
    async with engine.connect() as conn:
        result = await conn.execute(
            select(Chunk.text)
            .where(Chunk.document_version_id == version_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())
