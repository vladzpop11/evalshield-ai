"""API-level request byte and aggregate evaluation workload limits."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI, Request
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


@pytest.mark.asyncio
async def test_concurrent_body_admission_rejects_excess_and_recovers_after_success():
    limit = 2
    entered = 0
    entered_lock = asyncio.Lock()
    both_entered = asyncio.Event()
    release = asyncio.Event()
    inner_app = FastAPI()

    @inner_app.post("/{mode}")
    async def endpoint(mode: str, request: Request):
        nonlocal entered
        body = await request.body()
        if mode == "hold":
            async with entered_lock:
                entered += 1
                if entered == limit:
                    both_entered.set()
            await release.wait()
        return {"body_size": len(body)}

    middleware = api_main.RequestBodyLimitMiddleware(
        inner_app,
        max_body_bytes=8,
        max_concurrent_requests=limit,
    )
    async with AsyncClient(
        transport=ASGITransport(app=middleware), base_url="http://testserver"
    ) as client:
        admitted = [
            asyncio.create_task(client.post("/hold", content=b"one")),
            asyncio.create_task(client.post("/hold", content=b"two")),
        ]
        try:
            await asyncio.wait_for(both_entered.wait(), timeout=2)
            rejected = await asyncio.wait_for(
                client.post("/hold", content=b"excess"), timeout=1
            )
            assert rejected.status_code == 503
            assert rejected.json() == {
                "detail": "Too many requests are being processed. Retry shortly."
            }
            assert rejected.headers["retry-after"] == "1"
            assert rejected.headers["connection"] == "close"
            assert entered == limit
        finally:
            release.set()

        responses = await asyncio.gather(*admitted)
        assert [response.status_code for response in responses] == [200, 200]

        # Repeated requests after both completions prove no admission slots leaked.
        for _ in range(20):
            response = await client.post("/ok", content=b"ok")
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_body_admission_slots_release_after_exception_oversize_and_cancellation():
    cancel_started = asyncio.Event()
    never_release = asyncio.Event()
    inner_app = FastAPI()

    @inner_app.post("/{mode}")
    async def endpoint(mode: str, request: Request):
        body = await request.body()
        if mode == "raise":
            raise RuntimeError("controlled endpoint failure")
        if mode == "cancel":
            cancel_started.set()
            await never_release.wait()
        return {"body_size": len(body)}

    middleware = api_main.RequestBodyLimitMiddleware(
        inner_app,
        max_body_bytes=4,
        max_concurrent_requests=1,
    )
    async with AsyncClient(
        transport=ASGITransport(app=middleware, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        oversized = await client.post("/ok", content=b"12345")
        assert oversized.status_code == 413
        assert (await client.post("/ok", content=b"1234")).status_code == 200

        failed = await client.post("/raise", content=b"x")
        assert failed.status_code == 500
        assert (await client.post("/ok", content=b"x")).status_code == 200

        cancelled_request = asyncio.create_task(client.post("/cancel", content=b"x"))
        await asyncio.wait_for(cancel_started.wait(), timeout=2)
        cancelled_request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_request

        # The permit is released from finally even when downstream is cancelled.
        recovered = await client.post("/ok", content=b"x")
        assert recovered.status_code == 200


@pytest.mark.asyncio
async def test_disconnect_releases_body_admission_slot():
    async def responding_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = api_main.RequestBodyLimitMiddleware(
        responding_app,
        max_body_bytes=4,
        max_concurrent_requests=1,
    )
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/request",
        "raw_path": b"/request",
        "query_string": b"",
        "headers": [],
        "server": ("testserver", 80),
        "client": ("testclient", 1234),
    }

    async def disconnected_receive():
        return {"type": "http.disconnect"}

    async def request_receive():
        return {"type": "http.request", "body": b"x", "more_body": False}

    async def run_request(receive):
        sent = []

        async def send(message):
            sent.append(message)

        await middleware(scope, receive, send)
        return sent

    disconnected_response = await run_request(disconnected_receive)
    assert disconnected_response[0]["status"] == 200
    normal_response = await run_request(request_receive)
    assert normal_response[0]["status"] == 200


@pytest.mark.asyncio
async def test_full_admission_returns_safe_503_before_auth_and_keeps_health_available(
    api_client, monkeypatch
):
    monkeypatch.setitem(
        api_main.app.state._state,
        "api_auth_config",
        APIAuthConfig(mode="production", api_key="k" * 40),
    )
    # Build Starlette's middleware stack and locate the production limiter.
    assert (await api_client.get("/health")).status_code == 200
    current = api_main.app.middleware_stack
    body_limiter = None
    while current is not None:
        if isinstance(current, api_main.RequestBodyLimitMiddleware):
            body_limiter = current
            break
        current = getattr(current, "app", None)
    assert body_limiter is not None

    acquired = 0
    for _ in range(api_main.MAX_CONCURRENT_BUFFERED_REQUESTS):
        assert body_limiter._admission_slots.acquire(blocking=False)
        acquired += 1

    try:
        # Admission happens before route-level authentication, including for an
        # unauthenticated request, while the bodyless public health probe bypasses it.
        overloaded = await api_client.post("/evaluate", json=metric_item())
        health = await api_client.get("/health")
    finally:
        for _ in range(acquired):
            body_limiter._admission_slots.release()

    assert overloaded.status_code == 503
    assert overloaded.json() == {
        "detail": "Too many requests are being processed. Retry shortly."
    }
    assert overloaded.headers["retry-after"] == "1"
    assert overloaded.headers["x-content-type-options"] == "nosniff"
    assert overloaded.headers["x-frame-options"] == "DENY"
    assert health.status_code == 200

    # Once capacity is restored, the same request reaches auth and is rejected
    # with the existing 401 behavior rather than being admitted anonymously.
    unauthenticated = await api_client.post("/evaluate", json=metric_item())
    assert unauthenticated.status_code == 401
