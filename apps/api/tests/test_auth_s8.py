"""Seam S8: POST /auth/token + permission enforcement (API responses only).

Verified behavior (docs/03 seam map): tenant isolation, role denial, expiry —
all observed through API responses with auth_enabled=True.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from atlas_api.main import create_app
from atlas_core.auth import AuthError, create_access_token, decode_access_token
from atlas_core.config import Settings
from httpx import ASGITransport, AsyncClient, Response

if TYPE_CHECKING:
    from fastapi import FastAPI

    ClientPair = tuple[AsyncClient, "FastAPI"]
else:
    ClientPair = tuple

AUTH_SETTINGS = Settings(
    database_url="postgresql+asyncpg://atlas:atlas@localhost:5433/atlas",
    auth_enabled=True,
    jwt_secret="test-secret",
    token_ttl_seconds=3600,
)

CONTENT = "# Terms\n\nThe refund period is 42 days."


def _auth_app() -> FastAPI:
    return create_app(AUTH_SETTINGS)


def _bearer(app: FastAPI, tenant: str, role: str = "admin", ttl: int | None = None) -> str:
    settings: Settings = app.state.settings
    return create_access_token(
        tenant=tenant,
        role=role,
        secret=settings.jwt_secret,
        ttl_seconds=ttl if ttl is not None else settings.token_ttl_seconds,
    )


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# === unit: token mint/verify ===


def test_token_roundtrip_preserves_claims() -> None:
    token = create_access_token(tenant="acme", role="editor", secret="s", now=1000.0)
    claims = decode_access_token(token, secret="s", now=1000.0)
    assert claims.tenant == "acme"
    assert claims.role == "editor"
    assert claims.exp == 4600


def test_expired_token_rejected() -> None:
    token = create_access_token(tenant="t", role="employee", secret="s", ttl_seconds=10, now=1000.0)
    with pytest.raises(AuthError, match="expired"):
        decode_access_token(token, secret="s", now=1010.5)


def test_wrong_secret_rejected() -> None:
    token = create_access_token(tenant="t", role="employee", secret="right")
    with pytest.raises(AuthError, match="bad-signature"):
        decode_access_token(token, secret="wrong")


def test_tampered_payload_rejected() -> None:
    token = create_access_token(tenant="tenant-a", role="employee", secret="s")
    header, payload, sig = token.split(".")
    # deterministically change the FIRST char so the segment always differs
    forged_payload = ("A" if payload[0] != "A" else "B") + payload[1:]
    assert forged_payload != payload
    with pytest.raises(AuthError, match="bad-signature"):
        decode_access_token(f"{header}.{forged_payload}.{sig}", secret="s")


def test_malformed_tokens_rejected() -> None:
    for bad in ("", "not-a-jwt", "a.b"):
        with pytest.raises(AuthError, match="malformed"):
            decode_access_token(bad, secret="s")


def test_unknown_role_rejected_at_mint() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        create_access_token(tenant="t", role="superuser", secret="s")


# === API: enforcement ===


async def test_query_without_token_is_401() -> None:
    app = _auth_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post("/query", json={"question": "anything"})
    assert response.status_code == 401


async def test_query_with_expired_token_is_401() -> None:
    app = _auth_app()
    expired = _bearer(app, "acme", ttl=-1)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post(
            "/query", json={"question": "anything"}, headers=_headers(expired)
        )
    assert response.status_code == 401


async def test_query_with_tampered_token_is_401() -> None:
    app = _auth_app()
    token = _bearer(app, "acme")
    header, payload, sig = token.split(".")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post(
            "/query",
            json={"question": "anything"},
            headers=_headers(f"{header}.{payload[:-2]}AA.{sig}"),
        )
    assert response.status_code == 401


async def _upload_with(app: FastAPI, token: str, key: str) -> Response:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        return await http.post(
            "/documents",
            json={"title": "T", "doc_type": "policy", "content": CONTENT},
            headers={"Idempotency-Key": key, **_headers(token)},
        )


async def test_employee_cannot_upload_but_admin_can(client: ClientPair) -> None:
    _, legacy_app = client
    app = create_app(
        AUTH_SETTINGS.model_copy(update={"database_url": legacy_app.state.settings.database_url})
    )
    employee = _bearer(app, "acme", role="employee")
    admin = _bearer(app, "acme", role="admin")

    denied = await _upload_with(app, employee, "s8-denied")
    allowed = await _upload_with(app, admin, "s8-allowed")

    assert denied.status_code == 403
    assert allowed.status_code == 202


async def test_token_tenant_drives_isolation_end_to_end(client: ClientPair) -> None:
    """Docs uploaded under tenant-a are invisible to a tenant-b token holder."""
    _, legacy_app = client
    app = create_app(
        AUTH_SETTINGS.model_copy(update={"database_url": legacy_app.state.settings.database_url})
    )
    admin_a = _bearer(app, "tenant-a", role="admin")
    employee_b = _bearer(app, "tenant-b", role="employee")

    uploaded = await _upload_with(app, admin_a, "s8-iso")
    assert uploaded is not None and uploaded.status_code == 202

    # NOTE: no X-Tenant-ID spoofing possible - tenancy comes from the token.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post(
            "/query",
            json={"question": "What is the refund period?"},
            headers={**_headers(employee_b), "X-Tenant-ID": "tenant-a"},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["abstained"] is True  # tenant-b sees nothing, fail closed
    assert body["citations"] == []


async def test_health_stays_open_in_auth_mode() -> None:
    app = _auth_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.get("/health")
    assert response.status_code == 200


async def test_token_issuance_endpoint_roundtrip() -> None:
    app = _auth_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        issued = await http.post("/auth/token", json={"tenant": "acme", "role": "admin"})
        assert issued.status_code == 200
        body = issued.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == AUTH_SETTINGS.token_ttl_seconds

        # minted token actually works on a protected route
        query = await http.post(
            "/query", json={"question": "x"}, headers=_headers(body["access_token"])
        )
    assert query.status_code in (200, 503)  # auth passes; DB state decides the rest
