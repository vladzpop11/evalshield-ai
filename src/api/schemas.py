"""EvalShield-AI: API Pydantic Schemas (v2 compatible)."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from src.ml.model_generator import FEATURE_NAMES


MAX_BATCH_ITEMS = 100
MAX_BATCH_TEXT_CHARS = 512 * 1024
MAX_CANARY_TOKEN_CHARS = 256
MAX_DRIFT_SAMPLES_PER_FEATURE = 4_000
MAX_DRIFT_SAMPLES_TOTAL = 20_000

FEATURE_SAMPLE_RANGES = {
    "toxicity_score": (0.0, 1.0),
    "hallucination_risk": (0.0, 1.0),
    "prompt_injection_score": (0.0, 1.0),
    "semantic_drift": (0.0, 1.0),
    "token_entropy": (0.0, 10.0),
}


def _validate_feature_samples(samples: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Validate bounded, finite samples for the model's actual feature set."""
    total_samples = sum(len(values) for values in samples.values())
    if total_samples > MAX_DRIFT_SAMPLES_TOTAL:
        raise ValueError(f"At most {MAX_DRIFT_SAMPLES_TOTAL} samples are allowed per request")

    unknown_features = set(samples) - set(FEATURE_NAMES)
    if unknown_features:
        raise ValueError("Only model feature names are allowed")

    for feature_name, values in samples.items():
        if len(values) > MAX_DRIFT_SAMPLES_PER_FEATURE:
            raise ValueError(
                f"At most {MAX_DRIFT_SAMPLES_PER_FEATURE} samples are allowed per feature"
            )
        lower, upper = FEATURE_SAMPLE_RANGES[feature_name]
        if any(not math.isfinite(value) or value < lower or value > upper for value in values):
            raise ValueError(f"Samples for {feature_name} must be finite and within [{lower}, {upper}]")

    return samples


class EvaluationRequest(BaseModel):
    """Input payload for AI safety evaluation.

    Two operational modes:
    - **Text Mode (Preferred):** Supply ``prompt_text`` only. The embedded
      ``TextFeatureExtractor`` automatically derives all 5 risk dimensions in-line.
    - **Metric Mode (Programmatic):** Supply pre-computed float scores directly.
      Any metric explicitly provided overrides the auto-extracted value.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Text Mode — raw prompt only",
                    "value": {
                        "prompt_id": "req-001",
                        "prompt_text": "Summarize the quarterly revenue report.",
                        "response_text": "Quarterly revenue increased by 14% year over year.",
                    },
                },
                {
                    "summary": "Metric Mode — pre-computed scores",
                    "value": {
                        "prompt_id": "req-002",
                        "toxicity_score": 0.04,
                        "hallucination_risk": 0.12,
                        "prompt_injection_score": 0.02,
                        "semantic_drift": 0.15,
                        "token_entropy": 4.82,
                    },
                },
            ]
        }
    )

    prompt_id: Optional[str] = Field(default=None, max_length=128, description="Unique client-provided identifier")

    # Raw text inputs (enable Text Mode auto-extraction)
    prompt_text: Optional[str] = Field(default=None, max_length=65536, description="Raw prompt text for auto feature extraction")
    response_text: Optional[str] = Field(default=None, max_length=65536, description="Generated response text for alignment analysis")
    redact_pii: bool = Field(default=False, description="Scan and redact sensitive PII and API credentials")
    canary_tokens: Optional[List[str]] = Field(default=None, max_length=100, description="Honeytokens to detect system prompt exfiltration")

    # Explicit metric overrides (Metric Mode or partial overrides)
    toxicity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Toxicity probability [0.0–1.0]")
    hallucination_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Hallucination probability [0.0–1.0]")
    prompt_injection_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Injection attack probability [0.0–1.0]")
    semantic_drift: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Domain distance score [0.0–1.0]")
    token_entropy: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Token Shannon entropy [0.0–10.0]")

    @field_validator("prompt_text")
    @classmethod
    def validate_prompt_not_blank(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("prompt_text cannot be empty or whitespace only.")
        return v

    @field_validator("canary_tokens")
    @classmethod
    def validate_canary_token_lengths(cls, values: Optional[List[str]]) -> Optional[List[str]]:
        if values is not None and any(len(value) > MAX_CANARY_TOKEN_CHARS for value in values):
            raise ValueError(
                f"Each canary token must be at most {MAX_CANARY_TOKEN_CHARS} characters."
            )
        return values

    @model_validator(mode="after")
    def require_text_or_all_metrics(self) -> "EvaluationRequest":
        """Enforce that at least one of text mode or full metric mode inputs is provided."""
        has_text = self.prompt_text is not None and bool(self.prompt_text.strip())
        metrics = [self.toxicity_score, self.hallucination_risk, self.prompt_injection_score,
                   self.semantic_drift, self.token_entropy]
        has_all_metrics = all(m is not None for m in metrics)

        if not has_text and not has_all_metrics:
            raise ValueError(
                "Provide either 'prompt_text' (auto-extraction mode) or "
                "all 5 explicit metric scores (toxicity_score, hallucination_risk, "
                "prompt_injection_score, semantic_drift, token_entropy)."
            )
        return self


class PIIEntityResponse(BaseModel):
    """Safe metadata for a detected sensitive value; never contains the value."""
    entity_type: str
    start: int
    end: int
    confidence: float
    masked_placeholder: str
    redacted: bool
    source: Optional[Literal["prompt", "response"]] = None


class EvaluationResponse(BaseModel):
    """Output payload representing the shield decision and all metrics."""
    prompt_id: Optional[str] = None
    shield_decision: Literal["ALLOWED", "FLAGGED", "BLOCKED"] = Field(..., description="Safety gate verdict")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalized composite risk index [0.0–1.0]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    probabilities: Dict[str, float] = Field(..., description="Per-class probability distribution")
    violation_reasons: List[str] = Field(default_factory=list, description="Specific guardrail boundary breaches")
    category_scores: Dict[str, float] = Field(..., description="Resolved feature scores used for inference")
    risk_contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Explainable feature percentage contributions to composite risk score (sums to 1.0)",
    )
    pii_detected: List[PIIEntityResponse] = Field(
        default_factory=list,
        description="Detected sensitive PII entities or credentials",
    )
    sanitized_prompt: Optional[str] = Field(
        default=None,
        description="Sanitized text if redact_pii was enabled",
    )
    canary_leak_detected: bool = Field(
        default=False,
        description="Whether a system prompt canary token was leaked in the response text",
    )
    leaked_canary_token: Optional[str] = Field(
        default=None,
        description="Deprecated compatibility field; sensitive token values are never returned",
    )
    extraction_mode: str = Field(default="metric", description="'text' if auto-extracted, 'metric' if caller-provided")
    backend: str = Field(..., description="Inference execution backend (onnx | sklearn_joblib)")
    latency_ms: float = Field(..., description="End-to-end evaluation latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchEvaluationRequest(BaseModel):
    """Batch evaluation container payload."""
    items: List[EvaluationRequest] = Field(..., min_length=1, max_length=MAX_BATCH_ITEMS)

    @model_validator(mode="after")
    def validate_aggregate_text_size(self) -> "BatchEvaluationRequest":
        text_chars = sum(
            len(item.prompt_text or "") + len(item.response_text or "")
            for item in self.items
        )
        if text_chars > MAX_BATCH_TEXT_CHARS:
            raise ValueError(
                "Combined prompt_text and response_text in a batch must not exceed "
                f"{MAX_BATCH_TEXT_CHARS} characters."
            )
        return self


class BatchEvaluationResponse(BaseModel):
    """Batch evaluation response container."""
    results: List[EvaluationResponse]
    total_evaluated: int
    allowed_count: int
    flagged_count: int
    blocked_count: int
    batch_latency_ms: float


class HealthResponse(BaseModel):
    """Service health and readiness status."""
    status: Literal["healthy", "degraded", "unhealthy"]
    service_name: str = "EvalShield-AI"
    version: str = "1.3.0"
    uptime_seconds: float
    model_loaded: bool
    backend: str


class ModelMetadataResponse(BaseModel):
    """Detailed model metadata and training parameters."""
    backend: str
    feature_names: List[str]
    classes: List[str]
    model_metadata: Dict[str, Any]


class DriftCheckRequest(BaseModel):
    """Request payload containing sample feature distributions to compare against the baseline."""
    feature_samples: Dict[str, List[float]] = Field(
        ...,
        max_length=len(FEATURE_NAMES),
        description="Map of model feature names to bounded observed numeric value lists",
    )
    significance_level: float = Field(default=0.05, ge=0.001, le=0.20)

    @field_validator("feature_samples")
    @classmethod
    def validate_feature_samples(cls, samples: Dict[str, List[float]]) -> Dict[str, List[float]]:
        return _validate_feature_samples(samples)


class DriftFeatureResult(BaseModel):
    """Kolmogorov-Smirnov drift check result for a single feature."""
    feature_name: str
    drift_detected: bool
    ks_statistic: float
    p_value: float
    baseline_mean: float
    sample_mean: float


class DriftCheckResponse(BaseModel):
    """Aggregate drift detection summary across all monitored features."""
    overall_drift_detected: bool
    drifted_features_count: int
    total_features_evaluated: int
    features: List[DriftFeatureResult]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecalibrateRequest(BaseModel):
    """Payload to trigger closed-loop drift recalibration."""
    production_samples: Dict[str, List[float]] = Field(
        ...,
        max_length=len(FEATURE_NAMES),
        description="Recent bounded observed samples for model features",
    )
    significance_level: float = Field(default=0.05, ge=0.001, le=0.20)

    @field_validator("production_samples")
    @classmethod
    def validate_production_samples(cls, samples: Dict[str, List[float]]) -> Dict[str, List[float]]:
        return _validate_feature_samples(samples)


class RecalibrateResponse(BaseModel):
    """Detailed result of automated drift-triggered threshold recalibration."""
    recalibration_id: str
    recalibration_status: Literal["RECALIBRATED", "STABLE_NO_RECALIBRATION_NEEDED"]
    overall_drift_detected: bool
    features_recalibrated: List[str]
    previous_thresholds: Dict[str, Any]
    adjusted_thresholds: Dict[str, Any]
    canary_promotion_recommended: bool
    samples_evaluated: int
    applied_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ShadowStatsResponse(BaseModel):
    """Real-time Champion / Challenger shadow routing statistics."""
    champion_backend: str
    challenger_backend: Optional[str] = None
    challenger_active: bool
    shadow_traffic_rate: float
    risk_delta_threshold: float
    total_production_requests: int
    shadow_evaluations_run: int
    discrepancy_count: int
    decision_mismatches: int
    risk_delta_breaches: int
    decision_agreement_rate: float
    avg_champion_latency_ms: float
    avg_challenger_latency_ms: float
    recent_discrepancies: List[Dict[str, Any]] = Field(default_factory=list)


class ComplianceReportResponse(BaseModel):
    """Structured compliance export for NIST AI RMF 1.0 & EU AI Act Annex IV."""
    frameworks: List[str]
    report_generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    system_identification: Dict[str, Any]
    risk_classification: Dict[str, Any]
    model_architecture_and_specs: Dict[str, Any]
    adversarial_robustness_and_fuzzing: Dict[str, Any]
    data_governance_and_drift_controls: Dict[str, Any]
    explainability_and_decision_logic: Dict[str, Any]
    human_oversight_and_shadow_telemetry: Dict[str, Any]
    compliance_status: Literal["COMPLIANT", "PROVISIONALLY_COMPLIANT", "NON_COMPLIANT"]


class RedTeamSimulateRequest(BaseModel):
    """Configuration for an automated adversarial red-team campaign."""
    seed_prompt: str = Field(..., min_length=3, max_length=1000, description="Base adversarial seed prompt to mutate")
    rounds: int = Field(default=10, ge=1, le=50, description="Number of mutated variations to test")
    mutation_types: Optional[List[str]] = Field(
        default=None,
        description="Selective mutation operators (leet_speak, payload_split, base64_wrap, roleplay_frame, hypothetical, delimiter_injection)",
    )
    trigger_recalibration: bool = Field(
        default=True,
        description="Whether to trigger closed-loop drift recalibration if evasions are discovered",
    )


class RedTeamAttackVectorSchema(BaseModel):
    """Individual attack attempt data in red-team simulation."""
    vector_id: str
    mutation_type: str
    original_prompt: str
    mutated_prompt: str
    shield_decision: str
    risk_score: float
    evasion_succeeded: bool
    violation_reasons: List[str]
    latency_ms: float


class RedTeamSimulateResponse(BaseModel):
    """Detailed outcome and cryptographic verification of red-team campaign."""
    campaign_id: str
    seed_prompt: str
    total_mutations: int
    evasion_count: int
    attack_success_rate: float = Field(..., ge=0.0, le=1.0)
    attack_vectors: List[RedTeamAttackVectorSchema]
    discovered_bypasses: List[RedTeamAttackVectorSchema]
    cryptographic_receipt: str
    recalibration_triggered: bool
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RedTeamMutationsResponse(BaseModel):
    """Available genetic mutation strategies and descriptions."""
    available_mutations: Dict[str, str]

