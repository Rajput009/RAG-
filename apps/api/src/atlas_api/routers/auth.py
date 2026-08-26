"""Seam S8: POST /auth/token + bearer enforcement (dev-mode issuer).

The issuer accepts tenant+role directly (no user store yet - Phase 5 adds
credentials). What IS enforced for real: signed tokens, expiry, role gates,
and token-derived tenancy on every protected route.
"""

from atlas_core.auth import KNOWN_ROLES, TokenClaims, create_access_token, decode_access_token
from atlas_core.config import Settings
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenBody(BaseModel):
    tenant: str
    role: str = "employee"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/token", response_model=TokenResponse)
async def issue_token(body: TokenBody, request: Request) -> TokenResponse:
    settings: Settings = request.app.state.settings
    if body.tenant.strip() == "":
        raise HTTPException(status_code=422, detail="tenant must be non-empty")
    if body.role not in KNOWN_ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {sorted(KNOWN_ROLES)}")
    token = create_access_token(
        tenant=body.tenant,
        role=body.role,
        secret=settings.jwt_secret,
        ttl_seconds=settings.token_ttl_seconds,
    )
    return TokenResponse(access_token=token, expires_in=settings.token_ttl_seconds)


async def require_bearer(request: Request, authorization: str = Header(default="")) -> None:
    """Router dependency: 401 unless a valid bearer token is presented.

    On success the claims are stored on request.state.claims; protected
    handlers MUST derive tenancy from claims, never from client headers.
    """
    settings: Settings = request.app.state.settings
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        claims: TokenClaims = decode_access_token(credentials, secret=settings.jwt_secret)
    except Exception as exc:  # AuthError or unexpected parse failure -> 401
        raise HTTPException(status_code=401, detail=f"invalid token: {exc}") from exc
    request.state.claims = claims
