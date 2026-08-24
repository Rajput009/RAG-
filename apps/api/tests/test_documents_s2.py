"""Seam S2: POST /documents guarantees.

- Idempotency: one key => one upload/document/version, safe client retries.
- Atomic publication: partial failure never publishes; nothing half-searchable.
- Searchability contract: only status='published' versions are visible.
"""

import uuid
from typing import TYPE_CHECKING

import pytest
from atlas_core.db.models import Chunk, Document, DocumentVersion, Upload
from atlas_core.providers import EmbeddingResult, HashEmbeddingProvider
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

if TYPE_CHECKING:
    from fastapi import FastAPI

pytestmark = pytest.mark.usefixtures("db_engine")

ClientPair = tuple[AsyncClient, "FastAPI"]

CONTENT = (
    "The refund period for enterprise subscriptions is 30 days.\n\n"
    "Customers must provide 15 days notice before cancellation.\n\n"
    "Refunds are issued to the original payment method."
)


class FailingEmbedder(HashEmbeddingProvider):
    """Adapter-seam failure injector: raises when asked to embed beyond fail_index."""

    def __init__(self, fail_index: int) -> None:
        self.fail_index = fail_index

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        if len(texts) > self.fail_index:
            await super().embed(texts[: self.fail_index])
            raise RuntimeError("embedding provider unavailable")
        return await super().embed(texts)


async def _post(client: AsyncClient, key: str, content: str = CONTENT) -> dict[str, object]:
    response = await client.post(
        "/documents",
        json={"title": "Refund Policy", "doc_type": "policy", "content": content},
        headers={"Idempotency-Key": key},
    )
    return dict(response.json()) | {"status_code": response.status_code}


async def test_upload_reaches_published_with_all_chunks_persisted(client: ClientPair) -> None:
    http, app = client
    result = await _post(http, "happy-1")
    assert result["status_code"] == 202
    assert result["replayed"] is False

    document_id = uuid.UUID(str(result["document_id"]))
    version_id = uuid.UUID(str(result["version_id"]))

    doc_response = await http.get(f"/documents/{document_id}")
    body = doc_response.json()
    assert body["version_status"] == "published"
    assert body["status"] == "active"

    engine: AsyncEngine = app.state.engine
    async with engine.connect() as conn:
        count = (
            await conn.execute(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.document_version_id == version_id)
            )
        ).scalar_one()
    assert count == 3


async def test_same_idempotency_key_five_times_yields_single_version(client: ClientPair) -> None:
    http, app = client

    first = await _post(http, "retry-key")
    assert first["replayed"] is False
    for _ in range(4):
        again = await _post(http, "retry-key")
        assert again["replayed"] is True
        assert again["document_id"] == first["document_id"]
        assert again["version_id"] == first["version_id"]

    engine: AsyncEngine = app.state.engine
    async with engine.connect() as conn:
        upload_count = (await conn.execute(select(func.count()).select_from(Upload))).scalar_one()
        version_count = (
            await conn.execute(select(func.count()).select_from(DocumentVersion))
        ).scalar_one()
    assert upload_count == 1
    assert version_count == 1


async def test_missing_idempotency_key_is_rejected(client: ClientPair) -> None:
    http, _app = client
    response = await http.post("/documents", json={"title": "T", "content": "C"}, headers={})
    assert response.status_code == 422


async def test_embedding_failure_never_publishes_or_persists_chunks(client: ClientPair) -> None:
    from atlas_api.main import create_app
    from atlas_core.config import Settings
    from httpx import AsyncClient

    _, app = client
    failing_app = create_app(
        Settings(database_url=app.state.settings.database_url),
        embedding_provider=FailingEmbedder(fail_index=2),
    )
    async with AsyncClient(
        transport=ASGITransport(app=failing_app), base_url="http://test"
    ) as http:
        result = await _post(http, "failing-key")
        document_id = uuid.UUID(str(result["document_id"]))
        version_id = uuid.UUID(str(result["version_id"]))

        doc = (await http.get(f"/documents/{document_id}")).json()
        assert doc["version_status"] == "failed"

    engine: AsyncEngine = app.state.engine
    async with engine.connect() as conn:
        chunk_count = (
            await conn.execute(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.document_version_id == version_id)
            )
        ).scalar_one()
        status_row = (
            await conn.execute(
                select(Upload.status, Upload.error_detail).where(
                    Upload.idempotency_key == "failing-key"
                )
            )
        ).one()
    assert chunk_count == 0
    assert status_row.status == "failed"
    assert status_row.error_detail is not None


@pytest.mark.parametrize("fail_index", [0, 2])
async def test_redrive_after_failure_publishes_without_duplicate_chunks(
    client: ClientPair, fail_index: int
) -> None:
    from atlas_api.main import create_app
    from atlas_core.config import Settings
    from httpx import AsyncClient

    _, app = client
    db_url = app.state.settings.database_url
    failing_app = create_app(
        Settings(database_url=db_url),
        embedding_provider=FailingEmbedder(fail_index=fail_index),
    )
    async with AsyncClient(
        transport=ASGITransport(app=failing_app), base_url="http://test"
    ) as http:
        result = await _post(http, f"redrive-{fail_index}")
        version_id = uuid.UUID(str(result["version_id"]))
        document_id = uuid.UUID(str(result["document_id"]))

    healthy_app = create_app(Settings(database_url=db_url))
    async with AsyncClient(
        transport=ASGITransport(app=healthy_app), base_url="http://test"
    ) as http:
        redrive = await http.post(
            f"/documents/{document_id}/redrive",
            headers={"Idempotency-Key": f"redrive-{fail_index}"},
        )
        assert redrive.status_code == 202

        doc = (await http.get(f"/documents/{document_id}")).json()
        assert doc["version_status"] == "published"

    async with app.state.engine.connect() as conn:
        chunk_count = (
            await conn.execute(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.document_version_id == version_id)
            )
        ).scalar_one()
    assert chunk_count == 3


MULTI_SECTION_CONTENT = (
    "# Cancellation\n\n"
    "Customers must provide 20 days notice before cancellation.\n\n"
    "Cancellation takes effect at the end of the billing period.\n\n"
    "# Refunds\n\n"
    "The refund period for enterprise subscriptions is 30 days."
)


async def test_chunk_indexes_are_global_across_sections(client: ClientPair) -> None:
    from atlas_core.chunking import parse_markdown as _parse_sections

    sections = _parse_sections(MULTI_SECTION_CONTENT)
    assert [s.heading for s in sections.sections] == ["Cancellation", "Refunds"]

    http, app = client
    result = await _post(http, "sections-1", content=MULTI_SECTION_CONTENT)
    assert result["status_code"] == 202

    version_id = uuid.UUID(str(result["version_id"]))
    engine: AsyncEngine = app.state.engine
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(
                    Chunk.chunk_index,
                    Chunk.page_number,
                    Chunk.section_path,
                    Chunk.token_count,
                )
                .where(Chunk.document_version_id == version_id)
                .order_by(Chunk.chunk_index)
            )
        ).all()

    assert [r.chunk_index for r in rows] == [0, 1, 2]
    assert [r.page_number for r in rows] == [1, 1, 2]
    assert [r.section_path for r in rows] == [
        ["Cancellation"],
        ["Cancellation"],
        ["Refunds"],
    ]
    first_text_len = len("Customers must provide 20 days notice before cancellation.")
    assert rows[0].token_count == max(1, first_text_len // 4)


async def test_concurrent_same_key_posts_yield_single_version(client: ClientPair) -> None:
    import asyncio

    http, app = client
    key = "race-key"

    async def fire() -> dict[str, int | str]:
        response = await http.post(
            "/documents",
            json={"title": "T", "doc_type": "policy", "content": CONTENT},
            headers={"Idempotency-Key": key},
        )
        merged: dict[str, int | str] = {"code": response.status_code}
        merged.update(dict(response.json()))
        return merged

    results = await asyncio.gather(*(fire() for _ in range(5)))

    document_ids = {str(r["document_id"]) for r in results}
    assert len(document_ids) == 1
    assert all(r["code"] in (200, 202) for r in results)

    engine: AsyncEngine = app.state.engine
    async with engine.connect() as conn:
        upload_count = (await conn.execute(select(func.count()).select_from(Upload))).scalar_one()
        version_count = (
            await conn.execute(select(func.count()).select_from(DocumentVersion))
        ).scalar_one()
    assert upload_count == 1
    assert version_count == 1


async def test_concurrent_new_tenants_yield_single_org_each(client: ClientPair) -> None:
    import asyncio

    from atlas_core.db.models import Organization

    http, app = client

    async def fire(tenant: str) -> int:
        response = await http.post(
            "/documents",
            json={"title": "T", "doc_type": "policy", "content": CONTENT},
            headers={"Idempotency-Key": f"tenant-{tenant}", "X-Tenant-ID": tenant},
        )
        return response.status_code

    codes = await asyncio.gather(fire("concurrent-a"), fire("concurrent-a"), fire("concurrent-b"))
    assert all(code == 202 for code in codes)

    engine: AsyncEngine = app.state.engine
    async with engine.connect() as conn:
        names = (
            (
                await conn.execute(
                    select(Organization.name).where(
                        Organization.name.in_(["concurrent-a", "concurrent-b"])
                    )
                )
            )
            .scalars()
            .all()
        )
    assert sorted(names) == ["concurrent-a", "concurrent-b"]


async def test_same_content_different_key_is_deduplicated(client: ClientPair) -> None:
    http, app = client

    first = await _post(http, "dedup-first")
    assert first["status_code"] == 202

    second_response = await http.post(
        "/documents",
        json={"title": "Refund Policy Copy", "doc_type": "policy", "content": CONTENT},
        headers={"Idempotency-Key": "dedup-second"},
    )
    second = dict(second_response.json())
    assert second_response.status_code == 202
    assert second["deduplicated"] is True
    assert second["document_id"] == first["document_id"]
    assert second["version_id"] == first["version_id"]

    engine: AsyncEngine = app.state.engine
    async with engine.connect() as conn:
        doc_count = (await conn.execute(select(func.count()).select_from(Document))).scalar_one()
        upload_keys = (await conn.execute(select(Upload.idempotency_key))).scalars().all()
    assert doc_count == 1
    assert sorted(upload_keys) == ["dedup-first", "dedup-second"]


async def test_dedup_is_scoped_to_tenant(client: ClientPair) -> None:
    http, app = client

    await _post(http, "scoped-acme")
    other = await http.post(
        "/documents",
        json={"title": "T", "doc_type": "policy", "content": CONTENT},
        headers={"Idempotency-Key": "scoped-globex", "X-Tenant-ID": "globex"},
    )
    body = dict(other.json())
    assert other.status_code == 202
    assert body.get("deduplicated") is False

    engine: AsyncEngine = app.state.engine
    async with engine.connect() as conn:
        doc_count = (await conn.execute(select(func.count()).select_from(Document))).scalar_one()
    assert doc_count == 2
