"""EvalShield-AI: Automated Drift Recalibration Engine Tests.

Validates:
- Adaptive threshold adjustment on shifted production entropy
- Stability retention on identically distributed samples
- Closed-loop canary candidate model recommendations
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.drift_recalibration import (
    MAX_RECALIBRATION_HISTORY_ENTRIES,
    DriftRecalibrationEngine,
    get_recalibration_engine,
)


@pytest.fixture
def recal_engine() -> DriftRecalibrationEngine:
    engine = get_recalibration_engine()
    engine.reset_to_defaults()
    return engine


@pytest.mark.ml_quality
class TestDriftRecalibration:
    """Test suite for closed-loop statistical drift recalibration."""

    def test_stable_distribution_triggers_no_unnecessary_recalibration(
        self,
        recal_engine: DriftRecalibrationEngine,
    ):
        """Identically distributed production samples maintain stable thresholds."""
        rng = np.random.RandomState(42)
        baseline = recal_engine._get_baseline()

        # Sample directly from baseline distribution
        stable_samples = {
            "token_entropy": rng.choice(baseline["token_entropy"].values, size=60).tolist(),
            "toxicity_score": rng.choice(baseline["toxicity_score"].values, size=60).tolist(),
        }

        result = recal_engine.recalibrate(stable_samples, significance_level=0.01)
        assert result.overall_drift_detected is False
        assert result.recalibration_status == "STABLE_NO_RECALIBRATION_NEEDED"
        assert result.canary_promotion_recommended is False
        assert len(result.features_recalibrated) == 0

    def test_shifted_entropy_triggers_adaptive_threshold_recalibration(
        self,
        recal_engine: DriftRecalibrationEngine,
    ):
        """Drifted token entropy dynamically expands adaptive heuristic boundaries."""
        rng = np.random.RandomState(42)
        prev_upper = recal_engine.active_thresholds["entropy_upper"]

        # Shifted production samples with elevated entropy (e.g. code/technical domain shift)
        drifted_samples = {
            "token_entropy": rng.normal(loc=8.8, scale=0.3, size=80).tolist(),
        }

        result = recal_engine.recalibrate(drifted_samples, significance_level=0.05)
        assert result.overall_drift_detected is True
        assert result.recalibration_status == "RECALIBRATED"
        assert "token_entropy" in result.features_recalibrated
        assert result.canary_promotion_recommended is True

        # Adaptive threshold adjusted to accommodate production reality
        new_upper = result.adjusted_thresholds["entropy_upper"]
        assert new_upper >= prev_upper
        assert recal_engine.get_active_thresholds()["entropy_upper"] == new_upper

    def test_recalibration_history_tracking(
        self,
        recal_engine: DriftRecalibrationEngine,
    ):
        """Engine maintains sequential audit trail of all recalibration passes."""
        rng = np.random.RandomState(99)
        initial_history_len = len(recal_engine.recalibration_history)

        samples = {
            "token_entropy": rng.normal(loc=9.2, scale=0.2, size=50).tolist(),
        }
        res = recal_engine.recalibrate(samples)

        assert len(recal_engine.recalibration_history) == initial_history_len + 1
        assert recal_engine.recalibration_history[-1].recalibration_id == res.recalibration_id

    def test_recalibration_history_retains_only_newest_results(self, recal_engine):
        """History retention applies to all results, including empty and undersized inputs."""
        engine = DriftRecalibrationEngine(baseline_df=recal_engine._get_baseline())
        sample_marker = 7.123456789
        oldest = engine.recalibrate({"token_entropy": [sample_marker]})

        recent_results = []
        for index in range(MAX_RECALIBRATION_HISTORY_ENTRIES + 1):
            samples = {} if index % 2 == 0 else {"token_entropy": [sample_marker] * 14}
            result = engine.recalibrate(samples)
            assert result.recalibration_status == "STABLE_NO_RECALIBRATION_NEEDED"
            assert result.samples_evaluated == 0
            recent_results.append(result)

        history = engine.recalibration_history
        expected_ids = [
            result.recalibration_id
            for result in recent_results[-MAX_RECALIBRATION_HISTORY_ENTRIES:]
        ]
        assert isinstance(history, list)
        assert len(history) == MAX_RECALIBRATION_HISTORY_ENTRIES
        assert [result.recalibration_id for result in history] == expected_ids
        assert oldest.recalibration_id not in expected_ids

        result_data = history[-1].to_dict()
        assert set(result_data) == {
            "recalibration_id",
            "recalibration_status",
            "overall_drift_detected",
            "features_recalibrated",
            "previous_thresholds",
            "adjusted_thresholds",
            "canary_promotion_recommended",
            "samples_evaluated",
            "applied_at",
        }
        assert "production_samples" not in result_data
        assert sample_marker not in result_data.values()

        retained_ids = [result.recalibration_id for result in history]
        engine.reset_to_defaults()
        assert [result.recalibration_id for result in history] == retained_ids

        recalibrated = engine.recalibrate({"token_entropy": [9.8] * 80})
        assert recalibrated.recalibration_status == "RECALIBRATED"
        assert len(history) == MAX_RECALIBRATION_HISTORY_ENTRIES
        assert history[-1].recalibration_id == recalibrated.recalibration_id
