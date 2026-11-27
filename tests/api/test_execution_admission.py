"""Regression tests for bounded offloading of synchronous API work."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import main as api_main
from src.api.security import APIAuthConfig


def _metric_payload(prompt_id: str = "execution-test") -> dict:
    return {
        "prompt_id": prompt_id,
        "toxicity_score": 0.1,
        "hallucination_risk": 0.1,
        "prompt_injection_score": 0.1,
        "semantic_drift": 0.1,
        "token_entropy": 4.0,
    }


def _evaluation_result() -> tuple[dict, str]:
    scores = {
        "toxicity_score": 0.1,
        "hallucination_risk": 0.1,
        "prompt_injection_score": 0.1,
        "semantic_drift": 0.1,
        "token_entropy": 4.0,
    }
    return (
        {
            "prompt_id": "execution-test",
            "shield_decision": "ALLOWED",
            "risk_score": 0.1,
            "confidence": 0.9,
            "probabilities": {"ALLOWED": 0.9, "FLAGGED": 0.1, "BLOCKED": 0.0},
            "violation_reasons": [],
            "category_scores": scores,
            "risk_contributions": {name: 0.2 for name in scores},
            "pii_detected": [],
            "sanitized_prompt": None,
            "canary_leak_detected": False,
            "leaked_canary_token": None,
            "extraction_mode": "metric",
            "backend": "test",
            "latency_ms": 0.1,
        },
        "metric",
    )


async def _wait_thread_event(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 2), "worker did not reach synchronization point"


@pytest.mark.asyncio
async def test_executor_capacity_is_nonblocking_and_cancellation_keeps_slot_until_thread_finishes():
    executor = api_main.BoundedSyncExecutor(max_workers=1)
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def controlled_work() -> str:
        started.set()
        release.wait()
        return "complete"

    try:
        future = executor.try_submit(controlled_work)
        assert future is not None
        future.add_done_callback(lambda _future: completed.set())
        await _wait_thread_event(started)

        wait_task = asyncio.create_task(executor.wait(future))
        await asyncio.sleep(0)
        wait_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wait_task

        # The HTTP waiter is gone, but the worker is still running; no new work
        # can enter and no executor queue can accumulate behind it.
        assert executor.try_submit(lambda: "must not queue") is None
        release.set()
        assert await asyncio.to_thread(completed.wait, 2)
        assert future.result() == "complete"

        recovered = executor.try_submit(lambda: "recovered")
        assert recovered is not None
        assert await executor.wait(recovered) == "recovered"
    finally:
        release.set()
        executor.shutdown()


@pytest.mark.asyncio
async def test_executor_releases_capacity_after_worker_and_submission_failures(monkeypatch):
    executor = api_main.BoundedSyncExecutor(max_workers=1)

    def fail_work() -> None:
        raise RuntimeError("controlled failure")

    try:
        failed = executor.try_submit(fail_work)
        assert failed is not None
        with pytest.raises(RuntimeError, match="controlled failure"):
            await executor.wait(failed)

        recovered = executor.try_submit(lambda: "recovered")
        assert recovered is not None
        assert await executor.wait(recovered) == "recovered"

        real_submit = executor._executor.submit

        def fail_submission(*_args, **_kwargs):
            raise RuntimeError("executor unavailable")

        monkeypatch.setattr(executor._executor, "submit", fail_submission)
        assert executor.try_submit(lambda: None) is None
        monkeypatch.setattr(executor._executor, "submit", real_submit)
        after_submit_failure = executor.try_submit(lambda: "slot released")
        assert after_submit_failure is not None
        assert await executor.wait(after_submit_failure) == "slot released"
    finally:
        executor.shutdown()

    # A shut down executor rejects submission and does not retain a reservation.
    assert executor.try_submit(lambda: None) is None


@pytest.mark.asyncio
async def test_executor_releases_capacity_when_submitted_future_is_cancelled(monkeypatch):
    executor = api_main.BoundedSyncExecutor(max_workers=1)
    cancelled_future: Future[str] = Future()
    cancelled_future.cancel()
    real_submit = executor._executor.submit

    try:
        monkeypatch.setattr(
            executor._executor,
            "submit",
            lambda *_args, **_kwargs: cancelled_future,
        )
        assert executor.try_submit(lambda: "cancelled") is cancelled_future

        monkeypatch.setattr(executor._executor, "submit", real_submit)
        recovered = executor.try_submit(lambda: "recovered")
        assert recovered is not None
        assert await executor.wait(recovered) == "recovered"
    finally:
        executor.shutdown()


@pytest.mark.asyncio
async def test_expensive_routes_offload_and_reject_when_busy_without_blocking_health(
    monkeypatch,
):
    executor = api_main.BoundedSyncExecutor(max_workers=1)
    started = threading.Event()
    release = threading.Event()
    worker_thread_ids: list[int] = []
    api_key = "a" * 40
    monkeypatch.setitem(
        api_main.app.state._state,
        "api_auth_config",
        APIAuthConfig(mode="production", api_key=api_key),
    )
    monkeypatch.setattr(api_main, "get_expensive_executor", lambda: executor)
    monkeypatch.setattr(
        api_main,
        "get_engine",
        lambda: SimpleNamespace(backend="test"),
    )

    def controlled_evaluation(_payload):
        worker_thread_ids.append(threading.get_ident())
        started.set()
        release.wait()
        return _evaluation_result()

    monkeypatch.setattr(api_main, "_evaluate_interaction", controlled_evaluation)
    headers = {"Authorization": f"Bearer {api_key}"}
    loop_thread_id = threading.get_ident()

    try:
        async with AsyncClient(
            transport=ASGITransport(app=api_main.app), base_url="http://testserver"
        ) as client:
            active_request = asyncio.create_task(
                client.post("/evaluate", json=_metric_payload(), headers=headers)
            )
            await _wait_thread_event(started)

            assert len(worker_thread_ids) == 1
            assert worker_thread_ids[0] != loop_thread_id

            # Health stays responsive while the sole expensive worker is held.
            health = await asyncio.wait_for(client.get("/health"), timeout=2)
            assert health.status_code == 200

            busy_requests = [
                ("/evaluate", _metric_payload("busy-single")),
                ("/evaluate/batch", {"items": [_metric_payload("busy-batch")]}),
                ("/evaluate/stream", {"prompt_text": "safe stream input"}),
                ("/drift/check", {"feature_samples": {}}),
                ("/drift/recalibrate", {"production_samples": {}}),
                ("/red-team/simulate", {"seed_prompt": "safe seed", "rounds": 1}),
            ]
            for path, payload in busy_requests:
                response = await asyncio.wait_for(
                    client.post(path, json=payload, headers=headers), timeout=2
                )
                assert response.status_code == 503, path
                assert response.headers["retry-after"] == "1"
                assert "resource-intensive operation" in response.json()["detail"]

            release.set()
            completed = await asyncio.wait_for(active_request, timeout=2)
            assert completed.status_code == 200
            assert completed.json()["shield_decision"] == "ALLOWED"

            # Capacity is restored after actual worker completion; normal
            # authentication still rejects the same route without credentials.
            unauthenticated = await client.post("/evaluate", json=_metric_payload())
            assert unauthenticated.status_code == 401
            admitted_again = await client.post(
                "/evaluate", json=_metric_payload("recovered"), headers=headers
            )
            assert admitted_again.status_code == 200
    finally:
        release.set()
        executor.shutdown()
