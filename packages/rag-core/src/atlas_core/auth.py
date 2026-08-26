"""Auth primitives (seam S8): HS256 JWT mint/verify with zero dependencies.

Verified behavior at this seam:
- create_access_token / decode_access_token round-trip claims exactly.
- Failures are LOUD and typed: AuthError with reason (malformed | bad-signature |
  expired) - callers map them to 401, never silently accept.
- Signature comparison is constant-time (hmac.compare_digest).
- Roles are a closed set; document upload is restricted to UPLOAD_ROLES.
"""

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

ROLE_EMPLOYEE = "employee"
ROLE_EDITOR = "editor"
ROLE_ADMIN = "admin"
KNOWN_ROLES = frozenset({ROLE_EMPLOYEE, ROLE_EDITOR, ROLE_ADMIN})
# Upload/ingest mutates corpus state: restricted to elevated roles.
UPLOAD_ROLES = frozenset({ROLE_EDITOR, ROLE_ADMIN})


class AuthError(Exception):
    """Token rejected. `reason` in {malformed, bad-signature, expired}."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class TokenClaims:
    tenant: str
    role: str
    exp: int  # unix seconds


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def _encode_json(data: dict[str, Any]) -> str:
    return _b64url_encode(json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def create_access_token(
    *,
    tenant: str,
    role: str,
    secret: str,
    ttl_seconds: int = 3600,
    now: float | None = None,
) -> str:
    """Mint an HS256 JWT carrying tenant + role. ttl_seconds may be <= 0 (tests)."""
    if not tenant.strip():
        raise ValueError("tenant must be non-empty")
    if role not in KNOWN_ROLES:
        raise ValueError(f"unknown role: {role!r}")
    issued = int(now if now is not None else time.time())
    header = _encode_json({"alg": "HS256", "typ": "JWT"})
    payload = _encode_json(
        {"tenant": tenant, "role": role, "iat": issued, "exp": issued + ttl_seconds}
    )
    signing_input = f"{header}.{payload}"
    signature = hmac.new(
        secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def decode_access_token(token: str, *, secret: str, now: float | None = None) -> TokenClaims:
    """Verify signature + expiry and return claims. Raises AuthError."""
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("malformed")
    header_segment, payload_segment, signature_segment = parts

    expected = hmac.new(
        secret.encode("utf-8"),
        f"{header_segment}.{payload_segment}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        provided = _b64url_decode(signature_segment)
    except Exception as exc:
        raise AuthError("malformed") from exc
    if not hmac.compare_digest(provided, expected):
        raise AuthError("bad-signature")

    try:
        claims = json.loads(_b64url_decode(payload_segment))
        tenant, role, exp = claims["tenant"], claims["role"], int(claims["exp"])
    except Exception as exc:
        raise AuthError("malformed") from exc

    current = int(now if now is not None else time.time())
    if current >= exp:
        raise AuthError("expired")

    return TokenClaims(tenant=str(tenant), role=str(role), exp=exp)


def role_can_upload(role: str) -> bool:
    return role in UPLOAD_ROLES
