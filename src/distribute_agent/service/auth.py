"""JWT issuance and verification for the private media endpoint.

The endpoint is a resource server. Tokens are minted out of band by the
``token`` CLI command from a shared HS256 secret held in the service
environment, and FastMCP's native :class:`JWTVerifier` validates signature,
issuer, audience, expiry and the required scope on every MCP request.

FastMCP 4.1.3 checks ``exp`` only when the claim is present, so a hand-built
token with no expiry would be accepted. :func:`issue_token` always writes one;
do not mint tokens for this service with another tool.
"""

import os
import secrets
import time

from joserfc import jwt
from joserfc.jwk import OctKey

from .config import (
    DEFAULT_AUTH_AUDIENCE,
    DEFAULT_AUTH_ISSUER,
    DEFAULT_AUTH_SCOPE,
    DEFAULT_AUTH_SECRET_ENV,
    DEFAULT_AUTH_TOKEN_TTL_SECONDS,
)

MIN_SECRET_BYTES = 32
JWT_ALGORITHM = "HS256"


class AuthConfigError(ValueError):
    """The JWT settings cannot describe a usable verifier or token."""


def load_secret(*, env=None, secret_env: str = DEFAULT_AUTH_SECRET_ENV) -> str:
    """Return the shared secret, refusing a missing or short one."""
    source = os.environ if env is None else env
    secret = source.get(secret_env)
    if secret is None or not secret.strip():
        raise AuthConfigError(
            f"{secret_env} is not set; export a random secret of at least "
            f"{MIN_SECRET_BYTES} bytes before starting the service"
        )
    return validate_secret(secret, label=secret_env)


def validate_secret(secret: str, *, label: str = "JWT secret") -> str:
    if not isinstance(secret, str) or not secret.strip():
        raise AuthConfigError(f"{label} must be a non-empty string")
    if len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
        raise AuthConfigError(
            f"{label} must be at least {MIN_SECRET_BYTES} bytes; generate it with "
            "`python3 -c 'import secrets; print(secrets.token_urlsafe(48))'`"
        )
    return secret


def build_verifier(
    secret: str,
    *,
    issuer: str = DEFAULT_AUTH_ISSUER,
    audience: str = DEFAULT_AUTH_AUDIENCE,
    scope: str = DEFAULT_AUTH_SCOPE,
):
    """Build the native FastMCP verifier used by MCP and the artifact route."""
    validate_secret(secret)
    for label, value in (("issuer", issuer), ("audience", audience), ("scope", scope)):
        if not isinstance(value, str) or not value.strip():
            raise AuthConfigError(f"{label} must be a non-empty string")
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    try:
        return JWTVerifier(
            public_key=secret,
            algorithm=JWT_ALGORITHM,
            issuer=issuer,
            audience=audience,
            required_scopes=[scope],
        )
    except ValueError as exc:
        raise AuthConfigError(f"JWT verifier configuration rejected: {exc}") from exc


def issue_token(
    secret: str,
    *,
    subject: str = "owner",
    issuer: str = DEFAULT_AUTH_ISSUER,
    audience: str = DEFAULT_AUTH_AUDIENCE,
    scope: str = DEFAULT_AUTH_SCOPE,
    ttl_seconds: int = DEFAULT_AUTH_TOKEN_TTL_SECONDS,
    now=None,
) -> str:
    """Mint one HS256 access token with the claims the verifier requires."""
    validate_secret(secret)
    if not isinstance(subject, str) or not subject.strip():
        raise AuthConfigError("subject must be a non-empty string")
    for label, value in (("issuer", issuer), ("audience", audience), ("scope", scope)):
        if not isinstance(value, str) or not value.strip():
            raise AuthConfigError(f"{label} must be a non-empty string")
    if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
        raise AuthConfigError("ttl_seconds must be a positive integer")
    issued_at = int(time.time() if now is None else now)
    claims = {
        "sub": subject,
        "iss": issuer,
        "aud": audience,
        "scope": scope,
        "iat": issued_at,
        "exp": issued_at + ttl_seconds,
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(
        {"alg": JWT_ALGORITHM, "typ": "JWT"},
        claims,
        OctKey.import_key(secret),
        algorithms=[JWT_ALGORITHM],
    )
