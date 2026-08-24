"""Document ingestion endpoints (seam S2)."""

import uuid

from atlas_core.db.models import Document, DocumentVersion, Organization, Upload
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, update
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
        result = await session.execute(
            select(Organization).where(Organization.name == tenant_name)
        )
        org = result.scalar_one_or_none()
        if org is None:
            org = Organization(name=tenant_name)
            session.add(org)
            await session.commit()
            await session.refresh(org)
        assert isinstance(org, Organization)
        return org.id


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
        existing = (
            await session.execute(
                select(Upload).where(Upload.idempotency_key == idempotency_key)
            )
        ).scalar_one_or_none()
        if existing is not None:
            assert existing.document_id is not None and existing.version_id is not None
            version = await session.get(DocumentVersion, existing.version_id)
            return {
                "replayed": True,
                "document_id": str(existing.document_id),
                "version_id": str(existing.version_id),
                "status": version.status if version else "unknown",
            }

        document = Document(
            id=uuid.uuid4(),
            organization_id=org_id,
            title=body.title,
            doc_type=body.doc_type,
            status="processing",
        )
        version = DocumentVersion(
            id=uuid.uuid4(),
            document_id=document.id,
            version_number=1,
            content_hash=hash_value,
            effective_date="2026-01-01",
            status="indexing",
            source_text=body.content,
        )
        upload = Upload(
            id=uuid.uuid4(),
            idempotency_key=idempotency_key,
            content_hash=hash_value,
            status="pending",
            organization_id=org_id,
            document_id=document.id,
            version_id=version.id,
        )
        session.add_all([document, version, upload])
        await session.commit()

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
