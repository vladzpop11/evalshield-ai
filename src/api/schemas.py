"""EvalShield-AI: API Pydantic Schemas (v2 compatible)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class EvaluationRequest(BaseModel):
    """Input payload for AI safety and quality evaluation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt_id": "req-98765",
                "toxicity_score": 0.04,
                "hallucination_risk": 0.12,
                "prompt_injection_score": 0.02,
                "semantic_drift": 0.15,
                "token_entropy": 4.82,
                "prompt_text": "Summarize the quarterly revenue report.",
                "response_text": "Quarterly revenue increased by 14% year over year.",
            }
        }
    )

    prompt_id: Optional[str] = Field(default=None, description="Unique client-provided identifier")
    toxicity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Assessed toxicity probability [0.0 - 1.0]",
    )
    hallucination_risk: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Assessed hallucination probability [0.0 - 1.0]",
    )
    prompt_injection_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prompt injection attack probability [0.0 - 1.0]",
    )
    semantic_drift: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Distance from approved prompt domain embeddings [0.0 - 1.0]",
    )
    token_entropy: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Shannon entropy of the generated response token distribution",
    )
    prompt_text: Optional[str] = Field(default=None, description="Optional original prompt text")
    response_text: Optional[str] = Field(default=None, description="Optional generated response text")


class EvaluationResponse(BaseModel):
    """Output payload representing the shield decision and metrics."""
    prompt_id: Optional[str] = None
    shield_decision: Literal["ALLOWED", "FLAGGED", "BLOCKED"] = Field(
        ...,
        description="Safety gate verdict",
    )
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalized risk index [0.0 - 1.0]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probability distribution")
    violation_reasons: List[str] = Field(default_factory=list, description="Specific threshold breaches")
    category_scores: Dict[str, float] = Field(..., description="Echoed normalized input metric scores")
    backend: str = Field(..., description="Inference engine execution backend (onnx or sklearn)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
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
    """Request payload containing sample distributions to check against baseline."""
    feature_samples: Dict[str, List[float]] = Field(
        ...,
        description="Map of feature names to list of observed numeric values",
    )
    significance_level: float = Field(default=0.05, ge=0.001, le=0.20)


class DriftFeatureResult(BaseModel):
    """Drift check summary for a single feature."""
    feature_name: str
    drift_detected: bool
    ks_statistic: float
    p_value: float
    baseline_mean: float
    sample_mean: float


class DriftCheckResponse(BaseModel):
    """Summary of data drift detection across all monitored features."""
    overall_drift_detected: bool
    drifted_features_count: int
    total_features_evaluated: int
    features: List[DriftFeatureResult]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
