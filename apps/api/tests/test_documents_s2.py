"""Seam S2: POST /documents guarantees.

- Idempotency: one key => one upload/document/version, safe client retries.
- Atomic publication: partial failure never publishes; nothing half-searchable.
- Searchability contract: only status='published' versions are visible.
"""

import uuid
from typing import TYPE_CHECKING

import pytest
from atlas_core.db.models import Chunk, DocumentVersion, Upload
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


async def _post(
    client: AsyncClient, key: str, content: str = CONTENT
) -> dict[str, object]:
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
                select(func.count()).select_from(Chunk).where(
                    Chunk.document_version_id == version_id
                )
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
    response = await http.post(
        "/documents", json={"title": "T", "content": "C"}, headers={}
    )
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
                select(func.count()).select_from(Chunk).where(
                    Chunk.document_version_id == version_id
                )
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
                select(func.count()).select_from(Chunk).where(
                    Chunk.document_version_id == version_id
                )
            )
        ).scalar_one()
    assert chunk_count == 3
