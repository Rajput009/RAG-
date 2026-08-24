"""Document ingestion endpoints (seam S2)."""

import uuid

from atlas_core.db.models import Document, DocumentVersion, Organization, Upload
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from atlas_api.services.ingestion import content_hash, process_document

router = APIRouter(prefix="/documents", tags=["documents"])


class UploadBody(BaseModel):
    title: str
    doc_type: str = "policy"
    content: str


def _session(request: Request) -> AsyncSession:
    return async_sessionmaker(request.app.state.engine, expire_on_commit=False)()


async def _get_or_create_org_id(request: Request, tenant_name: str) -> uuid.UUID:
    async with _session(request) as session:
        await session.execute(
            pg_insert(Organization)
            .values(name=tenant_name)
            .on_conflict_do_nothing(index_elements=[Organization.name])
        )
        org_id = (
            await session.execute(
                select(Organization.id).where(Organization.name == tenant_name)
            )
        ).scalar_one()
        await session.commit()
        return org_id


async def _replay_response(session: AsyncSession, key: str) -> dict[str, object] | None:
    existing = (
        await session.execute(select(Upload).where(Upload.idempotency_key == key))
    ).scalar_one_or_none()
    if existing is None or existing.document_id is None or existing.version_id is None:
        return None
    version = await session.get(DocumentVersion, existing.version_id)
    return {
        "replayed": True,
        "deduplicated": False,
        "document_id": str(existing.document_id),
        "version_id": str(existing.version_id),
        "status": version.status if version else "unknown",
    }


@router.post("", status_code=202)
async def upload_document(
    body: UploadBody,
    request: Request,
    background_tasks: BackgroundTasks,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    x_tenant_id: str = Header(default="acme", alias="X-Tenant-ID"),
) -> dict[str, object]:
    if not idempotency_key.strip():
        raise HTTPException(status_code=422, detail="Idempotency-Key must be non-empty")

    org_id = await _get_or_create_org_id(request, x_tenant_id)
    hash_value = content_hash(body.content)

    async with _session(request) as session:
        replay = await _replay_response(session, idempotency_key)
        if replay is not None:
            return replay

        published_match = (
            await session.execute(
                select(DocumentVersion.id, DocumentVersion.document_id)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(
                    Document.organization_id == org_id,
                    DocumentVersion.content_hash == hash_value,
                    DocumentVersion.status == "published",
                )
                .limit(1)
            )
        ).first()

        document = Document(id=uuid.uuid4(), organization_id=org_id)
        version = DocumentVersion(id=uuid.uuid4())
        upload = Upload(id=uuid.uuid4(), idempotency_key=idempotency_key)

        if published_match is not None:
            upload.status = "completed"
            upload.organization_id = org_id
            upload.document_id = published_match.document_id
            upload.version_id = published_match.id
            upload.content_hash = hash_value
            session.add(upload)
            await session.commit()
            return {
                "replayed": False,
                "deduplicated": True,
                "document_id": str(published_match.document_id),
                "version_id": str(published_match.id),
                "status": "published",
            }

        document.title = body.title
        document.doc_type = body.doc_type
        document.status = "processing"
        version.document_id = document.id
        version.version_number = 1
        version.content_hash = hash_value
        version.effective_date = "2026-01-01"
        version.status = "indexing"
        version.source_text = body.content
        upload.content_hash = hash_value
        upload.status = "pending"
        upload.organization_id = org_id
        upload.document_id = document.id
        upload.version_id = version.id

        session.add_all([document, version, upload])
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            raced = await _replay_response(session, idempotency_key)
            if raced is not None:
                return raced
            raise HTTPException(status_code=409, detail="idempotency conflict") from None

    background_tasks.add_task(
        process_document,
        request.app.state.engine,
        request.app.state.embedding_provider,
        upload_id=upload.id,
        document_id=document.id,
        version_id=version.id,
        content=body.content,
    )

    return {
        "replayed": False,
        "deduplicated": False,
        "document_id": str(document.id),
        "version_id": str(version.id),
        "status": "indexing",
    }


@router.post("/{document_id}/redrive", status_code=202)
async def redrive_document(
    document_id: uuid.UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> dict[str, object]:
    """Re-drive a failed ingestion from the stored source text.

    Safe by construction against duplicate chunks: failed versions never had
    chunks committed, and publication is a single transaction.
    """
    async with _session(request) as session:
        upload = (
            await session.execute(
                select(Upload).where(Upload.idempotency_key == idempotency_key)
            )
        ).scalar_one_or_none()
        if upload is None or upload.document_id != document_id:
            raise HTTPException(status_code=404, detail="upload not found for key")
        version = await session.get(DocumentVersion, upload.version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="version not found")
        if version.status not in ("failed", "indexing"):
            raise HTTPException(status_code=409, detail=f"version is {version.status}")

        content = version.source_text
        await session.execute(
            update(DocumentVersion)
            .where(DocumentVersion.id == version.id)
            .values(status="indexing")
        )
        await session.execute(
            update(Upload)
            .where(Upload.id == upload.id)
            .values(status="pending", error_detail=None)
        )

    background_tasks.add_task(
        process_document,
        request.app.state.engine,
        request.app.state.embedding_provider,
        upload_id=upload.id,
        document_id=document_id,
        version_id=version.id,
        content=content,
    )
    return {"document_id": str(document_id), "version_id": str(version.id), "status": "indexing"}


@router.get("/{document_id}")
async def get_document(document_id: uuid.UUID, request: Request) -> dict[str, object]:
    async with _session(request) as session:
        document = await session.get(Document, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="document not found")
        version_status: str | None = None
        if document.current_version_id is not None:
            version = await session.get(DocumentVersion, document.current_version_id)
            version_status = version.status if version else None
        elif document.status == "processing":
            result = await session.execute(
                select(DocumentVersion).where(DocumentVersion.document_id == document.id)
            )
            first_version = result.scalars().first()
            version_status = first_version.status if first_version else "indexing"
        return {
            "document_id": str(document.id),
            "title": document.title,
            "status": document.status,
            "version_status": version_status,
        }
