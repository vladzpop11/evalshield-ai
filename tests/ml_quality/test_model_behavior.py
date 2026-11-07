"""EvalShield-AI: Model Behavioral Testing (CheckList methodology).

Implements:
1. Minimum Functionality Tests (MFT)
2. Invariance Tests (INV)
3. Directional Expectation Tests (DIR)
4. Guardrail and Safety Boundary Validation
"""

import pytest
from src.ml.inference import get_inference_engine


@pytest.fixture(scope="module")
def engine():
    """Shared fixture for inference engine."""
    return get_inference_engine()


@pytest.mark.ml_quality
class TestModelBehavior:
    """Rigorous behavioral evaluation of ML model decisions and safety shields."""

    # 1. Minimum Functionality Tests (MFT)
    def test_mft_benign_input_allowed(self, engine):
        """Sanity check: Clear benign inputs with low risks must be classified as ALLOWED."""
        benign_sample = {
            "toxicity_score": 0.02,
            "hallucination_risk": 0.08,
            "prompt_injection_score": 0.01,
            "semantic_drift": 0.05,
            "token_entropy": 4.5,
        }
        res = engine.predict(benign_sample)
        assert res["shield_decision"] == "ALLOWED"
        assert res["risk_score"] < 0.30
        assert len(res["violation_reasons"]) == 0

    def test_mft_severe_prompt_injection_blocked(self, engine):
        """A prompt injection attack vector with high confidence must be BLOCKED."""
        adversarial_sample = {
            "toxicity_score": 0.05,
            "hallucination_risk": 0.10,
            "prompt_injection_score": 0.92,
            "semantic_drift": 0.40,
            "token_entropy": 5.2,
        }
        res = engine.predict(adversarial_sample)
        assert res["shield_decision"] == "BLOCKED"
        assert res["risk_score"] >= 0.50
        assert any("injection" in v.lower() for v in res["violation_reasons"])

    def test_mft_severe_toxicity_blocked(self, engine):
        """Explicitly toxic inputs must be blocked unconditionally."""
        toxic_sample = {
            "toxicity_score": 0.88,
            "hallucination_risk": 0.15,
            "prompt_injection_score": 0.05,
            "semantic_drift": 0.20,
            "token_entropy": 4.2,
        }
        res = engine.predict(toxic_sample)
        assert res["shield_decision"] == "BLOCKED"
        assert any("toxicity" in v.lower() for v in res["violation_reasons"])

    # 2. Invariance Tests (INV)
    def test_invariance_under_minor_entropy_jitter(self, engine):
        """Invariance: Minor fluctuations in token entropy should not alter a benign ALLOWED decision."""
        base_sample = {
            "toxicity_score": 0.03,
            "hallucination_risk": 0.06,
            "prompt_injection_score": 0.02,
            "semantic_drift": 0.10,
            "token_entropy": 4.2,
        }
        base_decision = engine.predict(base_sample)["shield_decision"]
        assert base_decision == "ALLOWED"

        # Apply minor perturbations to entropy [4.0, 4.4, 4.6, 4.8]
        for perturbed_entropy in [4.0, 4.4, 4.6, 4.8]:
            sample = dict(base_sample, token_entropy=perturbed_entropy)
            perturbed_decision = engine.predict(sample)["shield_decision"]
            assert (
                perturbed_decision == "ALLOWED"
            ), f"Decision flipped to {perturbed_decision} with entropy {perturbed_entropy}"

    # 3. Directional Expectation Tests (DIR)
    def test_directional_risk_monotonicity_on_injection(self, engine):
        """Directional: Steadily increasing injection score must not decrease calculated risk score."""
        injection_levels = [0.05, 0.20, 0.45, 0.70, 0.90]
        prev_risk = -1.0

        for inj in injection_levels:
            sample = {
                "toxicity_score": 0.05,
                "hallucination_risk": 0.10,
                "prompt_injection_score": inj,
                "semantic_drift": 0.15,
                "token_entropy": 4.5,
            }
            res = engine.predict(sample)
            current_risk = res["risk_score"]
            # Allow for tiny numerical tolerance (0.02)
            assert (
                current_risk >= prev_risk - 0.02
            ), f"Risk decreased from {prev_risk} to {current_risk} when injection increased to {inj}"
            prev_risk = current_risk

    # 4. Latency Budget
    def test_inference_latency_budget(self, engine):
        """Verify inference latency satisfies production SLA (< 30ms single sample)."""
        sample = {
            "toxicity_score": 0.1,
            "hallucination_risk": 0.1,
            "prompt_injection_score": 0.1,
            "semantic_drift": 0.1,
            "token_entropy": 4.0,
        }
        res = engine.predict(sample)
        assert res["latency_ms"] < 30.0, f"Latency {res['latency_ms']}ms exceeded SLA of 30ms"
