"""EvalShield-AI: API Pydantic Schemas (v2 compatible)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    prompt_id: Optional[str] = Field(default=None, description="Unique client-provided identifier")

    # Raw text inputs (enable Text Mode auto-extraction)
    prompt_text: Optional[str] = Field(default=None, description="Raw prompt text for auto feature extraction")
    response_text: Optional[str] = Field(default=None, description="Generated response text for alignment analysis")
    redact_pii: bool = Field(default=False, description="Scan and redact sensitive PII and API credentials")
    canary_tokens: Optional[List[str]] = Field(default=None, description="Honeytokens to detect system prompt exfiltration")

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
    pii_detected: List[Dict[str, Any]] = Field(
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
        description="The specific leaked canary honeytoken if detected",
    )
    extraction_mode: str = Field(default="metric", description="'text' if auto-extracted, 'metric' if caller-provided")
    backend: str = Field(..., description="Inference execution backend (onnx | sklearn_joblib)")
    latency_ms: float = Field(..., description="End-to-end evaluation latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchEvaluationRequest(BaseModel):
    """Batch evaluation container payload."""
    items: List[EvaluationRequest] = Field(..., min_length=1, max_length=1000)


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
    version: str = "1.0.0"
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
        ..., description="Map of feature names to observed numeric value lists"
    )
    significance_level: float = Field(default=0.05, ge=0.001, le=0.20)


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
        ..., description="Recent observed production samples per feature"
    )
    significance_level: float = Field(default=0.05, ge=0.001, le=0.20)


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

