"""EvalShield-AI: Champion/Challenger Router Tests.

Validates:
- Synchronous zero-latency champion execution
- Asynchronous challenger shadow dispatch
- Telemetry counting and discrepancy logging
- Discrepancy JSONL persistence
"""

from __future__ import annotations

import json
import time
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
        tmp_path: Path,
    ):
        """Shadow evaluations run in background and increment shadow stats."""
        log_file = tmp_path / "test_discrepancies.jsonl"
        router = ChampionChallengerRouter(
            shadow_traffic_rate=1.0,
            discrepancy_log_path=log_file,
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
        tmp_path: Path,
    ):
        """When challenger disagrees with champion, discrepancy is logged and counted."""
        log_file = tmp_path / "discrepancies.jsonl"

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
            discrepancy_log_path=log_file,
        )

        res = router.route_and_evaluate(benign_features, prompt_id="discrepancy-req-1")
        # Champion verdict is authoritative
        assert res["shield_decision"] == "ALLOWED"

        time.sleep(0.3)

        stats = router.get_shadow_stats()
        assert stats["total_production_requests"] == 1
        assert stats["shadow_evaluations_run"] == 1
        assert stats["discrepancy_count"] == 1
        assert stats["decision_mismatches"] == 1
        assert stats["decision_agreement_rate"] == 0.0

        # Verify JSONL log was written
        assert log_file.exists()
        with open(log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 1
        assert lines[0]["prompt_id"] == "discrepancy-req-1"
        assert lines[0]["champion_decision"] == "ALLOWED"
        assert lines[0]["challenger_decision"] == "BLOCKED"
        assert lines[0]["decision_mismatch"] is True

        recent = router.get_recent_discrepancies()
        assert len(recent) == 1
        assert recent[0]["prompt_id"] == "discrepancy-req-1"

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
