"""Small environment-backed authentication configuration for the API."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from typing import Mapping, Optional

from fastapi import HTTPException, Request, status


@dataclass(frozen=True)
class APIAuthConfig:
    """API authentication policy. The configured key is intentionally repr-hidden."""

    mode: str
    api_key: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "APIAuthConfig":
        values = os.environ if environ is None else environ
        mode = values.get("EVALSHIELD_MODE", "demo").strip().lower()
        api_key = values.get("EVALSHIELD_API_KEY") or None
        return cls(mode=mode, api_key=api_key)

    @property
    def authentication_enabled(self) -> bool:
        # Unknown modes fail closed until startup validation rejects them.
        return self.mode != "demo" or self.api_key is not None

    def validate(self) -> "APIAuthConfig":
        """Validate mode and key policy without disclosing configured values."""
        if self.mode not in {"demo", "production"}:
            raise RuntimeError("EVALSHIELD_MODE must be either 'demo' or 'production'.")
        if self.mode == "production" and self.api_key is None:
            raise RuntimeError("EVALSHIELD_API_KEY is required when EVALSHIELD_MODE=production.")
        if self.api_key is not None and (
            len(self.api_key) < 32
            or not self.api_key.isascii()
            or any(character.isspace() for character in self.api_key)
        ):
            raise RuntimeError(
                "EVALSHIELD_API_KEY must be at least 32 non-whitespace ASCII characters."
            )
        return self


def documentation_urls(config: APIAuthConfig) -> dict[str, Optional[str]]:
    """Expose interactive API docs only when endpoints are in anonymous demo mode."""
    if config.authentication_enabled:
        return {"docs_url": None, "redoc_url": None, "openapi_url": None}
    return {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}


def valid_bearer_header(authorization: Optional[str], config: APIAuthConfig) -> bool:
    """Validate one Authorization Bearer credential using constant-time comparison."""
    if not config.api_key or not authorization or not authorization.isascii():
        return False
    scheme, separator, token = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not token
        or token != token.strip()
        or any(character.isspace() for character in token)
    ):
        return False
    return secrets.compare_digest(token, config.api_key)


async def require_api_auth(request: Request) -> None:
    """Central FastAPI dependency protecting every operation except public health."""
    config: Optional[APIAuthConfig] = getattr(request.app.state, "api_auth_config", None)
    if config is None:
        config = APIAuthConfig.from_env()
    if request.url.path == "/health" or not config.authentication_enabled:
        return
    if not valid_bearer_header(request.headers.get("Authorization"), config):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
