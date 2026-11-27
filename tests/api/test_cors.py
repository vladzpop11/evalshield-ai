"""Regression coverage for the API's explicit, non-credentialed CORS policy."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from src.api import main as api_main


def make_cors_test_app(origins: list[str]) -> FastAPI:
    test_app = FastAPI()
    api_main.add_api_cors_middleware(test_app, origins)

    @test_app.get("/probe")
    @test_app.post("/probe")
    async def probe():
        return {"ok": True}

    return test_app


def test_cors_origins_are_explicit_exact_origins_only():
    assert api_main.cors_origins_from_env({}) == []
    assert api_main.cors_origins_from_env(
        {"EVALSHIELD_CORS_ORIGINS": " http://localhost:8080,https://dashboard.example "}
    ) == ["http://localhost:8080", "https://dashboard.example"]

    for invalid in ("*", "https://*.example.com", "https://dashboard.example/path"):
        with pytest.raises(RuntimeError, match=r"exact http\(s\) origins"):
            api_main.cors_origins_from_env({"EVALSHIELD_CORS_ORIGINS": invalid})


def test_running_api_uses_configured_narrow_cors_policy():
    cors = next(
        middleware
        for middleware in api_main.app.user_middleware
        if middleware.cls is CORSMiddleware
    )
    assert cors.kwargs["allow_origins"] == api_main.cors_origins_from_env()
    assert cors.kwargs["allow_methods"] == ["GET", "POST"]
    assert cors.kwargs["allow_headers"] == ["Content-Type"]
    assert cors.kwargs["allow_credentials"] is False


@pytest.mark.asyncio
async def test_configured_origin_get_and_preflight_are_allowed_without_credentials():
    app = make_cors_test_app(["http://localhost:8080"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://api.test") as client:
        response = await client.get("/probe", headers={"Origin": "http://localhost:8080"})
        preflight = await client.options(
            "/probe",
            headers={
                "Origin": "http://localhost:8080",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8080"
    assert "access-control-allow-credentials" not in response.headers
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:8080"
    assert "access-control-allow-credentials" not in preflight.headers
    assert "POST" in preflight.headers["access-control-allow-methods"]
    assert "content-type" in preflight.headers["access-control-allow-headers"].lower()


@pytest.mark.asyncio
async def test_unconfigured_origin_and_preflight_are_denied():
    app = make_cors_test_app(["http://localhost:8080"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://api.test") as client:
        response = await client.get("/probe", headers={"Origin": "https://audit.invalid"})
        preflight = await client.options(
            "/probe",
            headers={
                "Origin": "https://audit.invalid",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers


@pytest.mark.asyncio
async def test_same_origin_dashboard_health_request_needs_no_cors_permission():
    async with AsyncClient(
        transport=ASGITransport(app=api_main.app), base_url="http://localhost:8080"
    ) as client:
        response = await client.get(
            "/health", headers={"Origin": "http://localhost:8080"}
        )

    assert response.status_code == 200
    assert response.json()["status"] in {"healthy", "degraded"}
