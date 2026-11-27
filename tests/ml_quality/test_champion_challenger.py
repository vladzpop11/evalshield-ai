"""EvalShield-AI: Champion/Challenger Router Tests.

Validates:
- Synchronous zero-latency champion execution
- Asynchronous challenger shadow dispatch
- Telemetry counting and safe structured discrepancy logging
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.ml.champion_challenger import ChampionChallengerRouter, get_champion_challenger_router
from src.ml.inference import EvalShieldEngine, get_inference_engine


@pytest.fixture
def benign_features() -> Dict[str, float]:
    return {
        "toxicity_score": 0.02,
        "hallucination_risk": 0.05,
        "prompt_injection_score": 0.01,
        "semantic_drift": 0.08,
        "token_entropy": 4.5,
    }


@pytest.fixture
def attack_features() -> Dict[str, float]:
    return {
        "toxicity_score": 0.10,
        "hallucination_risk": 0.20,
        "prompt_injection_score": 0.95,
        "semantic_drift": 0.65,
        "token_entropy": 5.0,
    }


@pytest.mark.ml_quality
class TestChampionChallengerRouter:
    """Test suite for Champion/Challenger shadow routing engine."""

    def test_singleton_retrieval(self):
        """get_champion_challenger_router returns a valid singleton."""
        router = get_champion_challenger_router()
        assert router is not None
        assert router.champion is not None
        stats = router.get_shadow_stats()
        assert "total_production_requests" in stats

    def test_synchronous_champion_routing(self, benign_features: Dict[str, float]):
        """Champion verdict is returned synchronously and adheres to schema."""
        router = ChampionChallengerRouter(shadow_traffic_rate=0.0)
        res = router.route_and_evaluate(benign_features, prompt_id="sync-test-01")

        assert "shield_decision" in res
        assert res["shield_decision"] == "ALLOWED"
        assert 0.0 <= res["risk_score"] <= 1.0
        assert res["backend"] in ["onnx_int8", "onnx", "sklearn_joblib"]
        router.shutdown()

    def test_shadow_evaluation_dispatched_asynchronously(
        self,
        benign_features: Dict[str, float],
    ):
        """Shadow evaluations run in background and increment shadow stats."""
        router = ChampionChallengerRouter(
            shadow_traffic_rate=1.0,
        )

        assert router.is_challenger_active is True

        for i in range(5):
            router.route_and_evaluate(benign_features, prompt_id=f"shadow-eval-{i}")

        # Allow background worker threads to finish
        time.sleep(0.4)

        stats = router.get_shadow_stats()
        assert stats["total_production_requests"] == 5
        assert stats["shadow_evaluations_run"] == 5
        assert 0.0 <= stats["decision_agreement_rate"] <= 1.0
        router.shutdown()

    def test_discrepancy_detection_and_logging(
        self,
        benign_features: Dict[str, float],
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Discrepancies are logged safely without writing to the artifact tree."""
        import logging

        caplog.set_level(logging.INFO, logger="evalshield_shadow_router")

        def reject_filesystem_write(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("discrepancy logging must not write files")

        monkeypatch.setattr(Path, "mkdir", reject_filesystem_write)

        mock_champion = MagicMock()
        mock_champion.backend = "onnx_int8"
        mock_champion.predict.return_value = {
            "shield_decision": "ALLOWED",
            "risk_score": 0.05,
            "confidence": 0.95,
            "backend": "onnx_int8",
            "latency_ms": 1.2,
        }

        mock_challenger = MagicMock()
        mock_challenger.backend = "candidate_v2"
        mock_challenger.predict.return_value = {
            "shield_decision": "BLOCKED",
            "risk_score": 0.88,
            "confidence": 0.90,
            "backend": "candidate_v2",
            "latency_ms": 1.8,
        }

        router = ChampionChallengerRouter(
            champion_engine=mock_champion,
            challenger_engine=mock_challenger,
            shadow_traffic_rate=1.0,
        )

        secret = "sk-" + "A" * 24
        sensitive_identifier = f"alice@example.com prompt {secret}"
        res = router.route_and_evaluate(benign_features, prompt_id=sensitive_identifier)
        # Champion verdict is authoritative
        assert res["shield_decision"] == "ALLOWED"

        time.sleep(0.3)

        stats = router.get_shadow_stats()
        assert stats["total_production_requests"] == 1
        assert stats["shadow_evaluations_run"] == 1
        assert stats["discrepancy_count"] == 1
        assert stats["decision_mismatches"] == 1
        assert stats["decision_agreement_rate"] == 0.0

        # The only observable discrepancy record is a JSON log without prompt IDs.
        log_messages = [r.getMessage() for r in caplog.records if "shadow_discrepancy=" in r.getMessage()]
        assert len(log_messages) == 1
        logged_record = json.loads(log_messages[0].split("=", 1)[1])
        assert logged_record["event"] == "shadow_discrepancy"
        assert logged_record["discrepancy_type"] == "decision_mismatch+risk_delta_breach"
        assert logged_record["champion_decision"] == "ALLOWED"
        assert logged_record["challenger_decision"] == "BLOCKED"
        assert logged_record["decision_mismatch"] is True
        assert logged_record["risk_delta"] == 0.83
        assert "prompt_id" not in logged_record
        assert sensitive_identifier not in log_messages[0]
        assert secret not in log_messages[0]
        assert "alice@example.com" not in log_messages[0]
        assert not any("Failed to append discrepancy log" in r.getMessage() for r in caplog.records)
        assert not hasattr(router, "discrepancy_log_path")

        recent = router.get_recent_discrepancies()
        assert len(recent) == 1
        assert recent[0]["prompt_id"] == sensitive_identifier

        router.shutdown()

    def test_zero_traffic_rate_dispatches_no_shadows(
        self,
        benign_features: Dict[str, float],
    ):
        """shadow_traffic_rate=0.0 routes only champion traffic."""
        router = ChampionChallengerRouter(shadow_traffic_rate=0.0)
        for i in range(3):
            router.route_and_evaluate(benign_features, prompt_id=f"rate-0-{i}")

        time.sleep(0.2)
        stats = router.get_shadow_stats()
        assert stats["total_production_requests"] == 3
        assert stats["shadow_evaluations_run"] == 0
        router.shutdown()

    def test_stats_reset(self, benign_features: Dict[str, float]):
        """reset_stats clears all counters and ring buffer."""
        router = ChampionChallengerRouter(shadow_traffic_rate=0.0)
        router.route_and_evaluate(benign_features, prompt_id="reset-test")
        stats = router.get_shadow_stats()
        assert stats["total_production_requests"] == 1

        router.reset_stats()
        stats_after = router.get_shadow_stats()
        assert stats_after["total_production_requests"] == 0
        assert stats_after["shadow_evaluations_run"] == 0
        router.shutdown()

    def test_shadow_work_is_bounded_and_saturation_preserves_champion(
        self,
        benign_features: Dict[str, float],
    ):
        """Saturation skips shadow jobs without delaying or changing champion results."""
        from src.ml.champion_challenger import DEFAULT_MAX_OUTSTANDING_SHADOW_JOBS

        worker_started = threading.Event()
        release_worker = threading.Event()
        callbacks_lock = threading.Lock()
        callbacks_completed = 0
        initial_callbacks_completed = threading.Event()
        third_callback_completed = threading.Event()

        champion = MagicMock()
        champion.backend = "onnx_int8"
        champion.predict.return_value = {
            "shield_decision": "ALLOWED",
            "risk_score": 0.05,
            "backend": "onnx_int8",
            "latency_ms": 1.0,
        }
        challenger = MagicMock()
        challenger.backend = "onnx"

        def blocked_prediction(_features: Dict[str, float]) -> Dict[str, Any]:
            worker_started.set()
            assert release_worker.wait(timeout=3)
            return {"shield_decision": "ALLOWED", "risk_score": 0.05, "latency_ms": 2.0}

        challenger.predict.side_effect = blocked_prediction
        router = ChampionChallengerRouter(
            champion_engine=champion,
            challenger_engine=challenger,
            shadow_traffic_rate=1.0,
            max_workers=1,
            max_outstanding_shadow_jobs=2,
        )
        original_done_callback = router._on_shadow_job_done

        def observe_completion(future: Future[Any]) -> None:
            nonlocal callbacks_completed
            original_done_callback(future)
            with callbacks_lock:
                callbacks_completed += 1
                if callbacks_completed >= 2:
                    initial_callbacks_completed.set()
                if callbacks_completed >= 3:
                    third_callback_completed.set()

        router._on_shadow_job_done = observe_completion  # type: ignore[method-assign]

        try:
            first = router.route_and_evaluate(benign_features, prompt_id="first")
            assert first["shield_decision"] == "ALLOWED"
            assert worker_started.wait(timeout=2)

            second = router.route_and_evaluate(benign_features, prompt_id="second")
            assert second["shield_decision"] == "ALLOWED"

            # More attempts than the cap are immediately skipped while one job
            # runs and one waits in the executor's queue.
            for index in range(DEFAULT_MAX_OUTSTANDING_SHADOW_JOBS * 2):
                result = router.route_and_evaluate(benign_features, prompt_id=f"saturated-{index}")
                assert result["shield_decision"] == "ALLOWED"
                stats = router.get_shadow_stats()
                assert stats["shadow_jobs_outstanding"] <= 2

            saturated_stats = router.get_shadow_stats()
            assert saturated_stats["shadow_jobs_outstanding"] == 2
            assert saturated_stats["shadow_jobs_skipped"] == DEFAULT_MAX_OUTSTANDING_SHADOW_JOBS * 2
            assert champion.predict.call_count == 2 + DEFAULT_MAX_OUTSTANDING_SHADOW_JOBS * 2

            release_worker.set()
            assert initial_callbacks_completed.wait(timeout=3)
            assert router.get_shadow_stats()["shadow_jobs_outstanding"] == 0

            # Capacity is available again after completion.
            after_release = router.route_and_evaluate(benign_features, prompt_id="after-release")
            assert after_release["shield_decision"] == "ALLOWED"
            assert third_callback_completed.wait(timeout=3)
            assert router.get_shadow_stats()["shadow_jobs_outstanding"] == 0
        finally:
            release_worker.set()
            router.shutdown(wait=True)

    def test_shadow_capacity_is_restored_after_worker_failure(
        self,
        benign_features: Dict[str, float],
    ):
        """A failed submitted future releases its admission slot."""
        completed = threading.Event()
        champion = MagicMock()
        champion.backend = "onnx_int8"
        champion.predict.return_value = {
            "shield_decision": "ALLOWED",
            "risk_score": 0.05,
            "backend": "onnx_int8",
            "latency_ms": 1.0,
        }
        challenger = MagicMock()
        challenger.backend = "onnx"
        challenger.predict.side_effect = RuntimeError("controlled worker failure")
        router = ChampionChallengerRouter(
            champion_engine=champion,
            challenger_engine=challenger,
            shadow_traffic_rate=1.0,
            max_outstanding_shadow_jobs=1,
        )
        original_done_callback = router._on_shadow_job_done

        def observe_completion(future: Future[Any]) -> None:
            original_done_callback(future)
            completed.set()

        router._on_shadow_job_done = observe_completion  # type: ignore[method-assign]

        try:
            result = router.route_and_evaluate(benign_features, prompt_id="failure")
            assert result["shield_decision"] == "ALLOWED"
            assert completed.wait(timeout=2)
            stats = router.get_shadow_stats()
            assert stats["shadow_jobs_outstanding"] == 0
            assert stats["shadow_jobs_skipped"] == 0
            assert challenger.predict.call_count == 1

            completed.clear()
            challenger.predict.side_effect = None
            challenger.predict.return_value = {
                "shield_decision": "ALLOWED",
                "risk_score": 0.05,
                "latency_ms": 2.0,
            }
            retry_result = router.route_and_evaluate(benign_features, prompt_id="after-failure")
            assert retry_result["shield_decision"] == "ALLOWED"
            assert completed.wait(timeout=2)
            assert challenger.predict.call_count == 2
            assert router.get_shadow_stats()["shadow_jobs_outstanding"] == 0
            assert router.get_shadow_stats()["shadow_jobs_skipped"] == 0
        finally:
            router.shutdown(wait=True)

    def test_shadow_capacity_is_restored_after_cancellation(
        self,
        benign_features: Dict[str, float],
    ):
        """Cancellation of a queued shadow future releases its reserved slot."""

        class PendingExecutor:
            def __init__(self) -> None:
                self.future: Future[None] = Future()

            def submit(self, *_args: Any, **_kwargs: Any) -> Future[None]:
                return self.future

            def shutdown(self, **_kwargs: Any) -> None:
                return None

        champion = MagicMock()
        champion.backend = "onnx_int8"
        champion.predict.return_value = {
            "shield_decision": "ALLOWED",
            "risk_score": 0.05,
            "backend": "onnx_int8",
            "latency_ms": 1.0,
        }
        challenger = MagicMock()
        challenger.backend = "onnx"
        router = ChampionChallengerRouter(
            champion_engine=champion,
            challenger_engine=challenger,
            shadow_traffic_rate=1.0,
            max_outstanding_shadow_jobs=1,
        )
        router._executor.shutdown(wait=True)
        pending_executor = PendingExecutor()
        router._executor = pending_executor  # type: ignore[assignment]

        try:
            result = router.route_and_evaluate(benign_features, prompt_id="cancel")
            assert result["shield_decision"] == "ALLOWED"
            assert router.get_shadow_stats()["shadow_jobs_outstanding"] == 1
            assert pending_executor.future.cancel()
            assert router.get_shadow_stats()["shadow_jobs_outstanding"] == 0
        finally:
            router.shutdown(wait=True)

    def test_shadow_submission_failure_releases_capacity_and_preserves_champion(
        self,
        benign_features: Dict[str, float],
    ):
        """Executor shutdown races do not fail the authoritative evaluation."""
        champion = MagicMock()
        champion.backend = "onnx_int8"
        champion.predict.return_value = {
            "shield_decision": "ALLOWED",
            "risk_score": 0.05,
            "backend": "onnx_int8",
            "latency_ms": 1.0,
        }
        challenger = MagicMock()
        challenger.backend = "onnx"
        router = ChampionChallengerRouter(
            champion_engine=champion,
            challenger_engine=challenger,
            shadow_traffic_rate=1.0,
            max_outstanding_shadow_jobs=1,
        )
        router.shutdown(wait=True)

        result = router.route_and_evaluate(benign_features, prompt_id="submit-failure")
        stats = router.get_shadow_stats()
        assert result["shield_decision"] == "ALLOWED"
        assert stats["shadow_jobs_outstanding"] == 0
        assert stats["shadow_jobs_skipped"] == 1
