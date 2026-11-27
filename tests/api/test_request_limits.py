"""API-level request byte and aggregate evaluation workload limits."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import main as api_main
from src.api.schemas import MAX_BATCH_ITEMS, MAX_BATCH_TEXT_CHARS, MAX_CANARY_TOKEN_CHARS
from src.api.security import APIAuthConfig


@pytest.fixture
def demo_api_auth(monkeypatch):
    monkeypatch.setitem(api_main.app.state._state, "api_auth_config", APIAuthConfig(mode="demo"))


@pytest.fixture
async def api_client(demo_api_auth):
    async with AsyncClient(
        transport=ASGITransport(app=api_main.app), base_url="http://testserver"
    ) as client:
        yield client


def metric_item(prompt_id: str = "limit-test") -> dict:
    return {
        "prompt_id": prompt_id,
        "toxicity_score": 0.1,
        "hallucination_risk": 0.1,
        "prompt_injection_score": 0.1,
        "semantic_drift": 0.1,
        "token_entropy": 4.0,
    }


@pytest.mark.asyncio
async def test_content_length_over_limit_returns_413_with_security_headers(api_client):
    request = api_client.build_request(
        "POST",
        "/evaluate",
        content=b" " * (api_main.MAX_REQUEST_BODY_BYTES + 1),
        headers={"Content-Type": "application/json"},
    )
    assert int(request.headers["content-length"]) == api_main.MAX_REQUEST_BODY_BYTES + 1
    response = await api_client.send(request)

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the 1 MiB limit."}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


@pytest.mark.asyncio
async def test_chunked_oversized_body_without_content_length_returns_413(api_client):
    async def chunks():
        yield b" " * (api_main.MAX_REQUEST_BODY_BYTES // 2)
        yield b" " * (api_main.MAX_REQUEST_BODY_BYTES // 2 + 1)

    request = api_client.build_request(
        "POST",
        "/evaluate",
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert "content-length" not in request.headers

    response = await api_client.send(request)

    assert response.status_code == 413


@pytest.mark.asyncio
async def test_body_at_exact_byte_limit_and_single_evaluation_remain_valid(api_client):
    payload = json.dumps({"prompt_text": "A representative safe request."}).encode()
    body = payload + b" " * (api_main.MAX_REQUEST_BODY_BYTES - len(payload))

    response = await api_client.post(
        "/evaluate", content=body, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    assert response.json()["extraction_mode"] == "text"


@pytest.mark.asyncio
async def test_batch_item_count_above_limit_is_rejected(api_client):
    response = await api_client.post(
        "/evaluate/batch",
        json={"items": [metric_item(str(index)) for index in range(MAX_BATCH_ITEMS + 1)]},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request payload."}


@pytest.mark.asyncio
async def test_aggregate_batch_text_above_limit_is_rejected(api_client):
    item_text = "x" * 60_000
    items = [{"prompt_text": item_text} for _ in range(9)]
    assert sum(len(item["prompt_text"]) for item in items) > MAX_BATCH_TEXT_CHARS

    response = await api_client.post("/evaluate/batch", json={"items": items})

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request payload."}


@pytest.mark.asyncio
async def test_canary_entry_above_length_limit_is_rejected(api_client):
    response = await api_client.post(
        "/evaluate",
        json={
            "prompt_text": "A representative safe request.",
            "canary_tokens": ["x" * (MAX_CANARY_TOKEN_CHARS + 1)],
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request payload."}


@pytest.mark.asyncio
async def test_valid_batch_within_aggregate_limits_remains_available(api_client):
    response = await api_client.post(
        "/evaluate/batch",
        json={"items": [metric_item("batch-1"), metric_item("batch-2")]},
    )

    assert response.status_code == 200
    assert response.json()["total_evaluated"] == 2


@pytest.mark.asyncio
async def test_body_limit_does_not_remove_authentication_or_public_health(monkeypatch):
    key = "k" * 40
    monkeypatch.setitem(
        api_main.app.state._state,
        "api_auth_config",
        APIAuthConfig(mode="production", api_key=key),
    )
    async with AsyncClient(
        transport=ASGITransport(app=api_main.app), base_url="http://testserver"
    ) as client:
        protected = await client.post("/evaluate", json=metric_item())
        health = await client.get("/health")

    assert protected.status_code == 401
    assert health.status_code == 200
