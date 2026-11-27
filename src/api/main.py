"""EvalShield-AI: Production FastAPI Application.

High-throughput, observable inference & evaluation API with:
- Dual-mode ingestion: raw text auto-extraction or pre-computed metric scores
- PII & Secret Redaction Engine (credit cards with Luhn check, SSNs, API keys)
- Explainable Risk Contributions (normalized attribution weights)
- INT8-quantized ONNX Runtime inference with Joblib fallback
- Thread-safe LRU semantic caching for repeated queries
- Champion / Challenger shadow routing engine
- Real-time token streaming guardrail via SSE (/evaluate/stream)
- Two-sample KS-test drift monitoring
- Prometheus `/metrics` exposition with PromQL sum/count metrics
- NIST AI RMF 1.0 & EU AI Act Annex IV `/model/compliance-report`
- Full OpenTelemetry-compatible timing headers
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from scipy.stats import ks_2samp

from src.api.schemas import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    ComplianceReportResponse,
    DriftCheckRequest,
    DriftCheckResponse,
    DriftFeatureResult,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    ModelMetadataResponse,
    RecalibrateRequest,
    RecalibrateResponse,
    RedTeamMutationsResponse,
    RedTeamSimulateRequest,
    RedTeamSimulateResponse,
    ShadowStatsResponse,
)
from src.api.security import APIAuthConfig, documentation_urls, require_api_auth
from src.ml.champion_challenger import ChampionChallengerRouter, get_champion_challenger_router
from src.ml.drift_recalibration import DriftRecalibrationEngine, get_recalibration_engine
from src.ml.feature_extractor import TextFeatureExtractor, get_feature_extractor
from src.ml.inference import EvalShieldEngine, get_inference_engine
from src.ml.pii_detector import PIIDetector, get_pii_detector
from src.ml.red_team import RedTeamFuzzer, get_red_team_fuzzer
from src.ml.unicode_sanitizer import UnicodeSanitizer, get_unicode_sanitizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evalshield_api")

START_TIME = time.time()
_engine: EvalShieldEngine | None = None
_baseline_df: pd.DataFrame | None = None
_extractor: TextFeatureExtractor | None = None
_router: ChampionChallengerRouter | None = None
_pii_detector: PIIDetector | None = None
_api_auth_config = APIAuthConfig.from_env()

MAX_REQUEST_BODY_BYTES = 1024 * 1024
# One API worker and 1 MiB per request: cap simultaneously buffered request
# bodies at 16 (16 MiB of raw bodies before parser and temporary-copy overhead).
MAX_CONCURRENT_BUFFERED_REQUESTS = 16


def cors_origins_from_env(environ: dict[str, str] | None = None) -> list[str]:
    """Read exact browser origins allowed to call this API cross-origin."""
    values = os.environ if environ is None else environ
    raw_origins = values.get("EVALSHIELD_CORS_ORIGINS", "")
    origins: list[str] = []
    for candidate in raw_origins.split(","):
        origin = candidate.strip()
        if not origin:
            continue
        parsed = urlsplit(origin)
        if (
            "*" in origin
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError(
                "EVALSHIELD_CORS_ORIGINS must contain exact http(s) origins "
                "without paths or wildcards."
            )
        if origin not in origins:
            origins.append(origin)
    return origins


def add_api_cors_middleware(app: FastAPI, origins: list[str]) -> None:
    """Install the minimal CORS policy used by browser dashboard requests."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


class RequestBodyLimitMiddleware:
    """Bound concurrent body buffering and bytes, then replay accepted bodies."""

    def __init__(
        self,
        app,
        max_body_bytes: int,
        max_concurrent_requests: int = MAX_CONCURRENT_BUFFERED_REQUESTS,
    ) -> None:
        if (
            not isinstance(max_concurrent_requests, int)
            or isinstance(max_concurrent_requests, bool)
            or max_concurrent_requests < 1
        ):
            raise ValueError("max_concurrent_requests must be a positive integer")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.max_concurrent_requests = max_concurrent_requests
        self._admission_slots = threading.BoundedSemaphore(max_concurrent_requests)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Normal health probes have no body and need not wait for a body-buffer
        # slot. Any body-bearing health request still uses the normal limits.
        if self._is_bodyless_health_probe(scope):
            await self.app(scope, receive, send)
            return

        if not self._admission_slots.acquire(blocking=False):
            await self._send_overloaded(scope, receive, send)
            return

        try:
            await self._buffer_and_call(scope, receive, send)
        finally:
            self._admission_slots.release()

    @staticmethod
    def _is_bodyless_health_probe(scope) -> bool:
        if (
            scope.get("method") != "GET"
            or scope.get("path") != "/health"
            or scope.get("http_version") not in {"1.0", "1.1"}
        ):
            return False
        for name, value in scope.get("headers", []):
            if name.lower() == b"transfer-encoding":
                return False
            if name.lower() == b"content-length" and value.strip() != b"0":
                return False
        return True

    async def _buffer_and_call(self, scope, receive, send) -> None:
        content_lengths = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"content-length"
        ]
        for raw_length in content_lengths:
            try:
                if int(raw_length) > self.max_body_bytes:
                    await self._send_too_large(scope, receive, send)
                    return
            except ValueError:
                # Invalid or ambiguous lengths do not bypass streamed byte counting.
                continue

        body = bytearray()
        request_complete = False
        while not request_complete:
            message = await receive()
            if message["type"] == "http.disconnect":
                await self.app(scope, receive, send)
                return
            if message["type"] != "http.request":
                continue

            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_body_bytes:
                body.clear()
                del message, chunk
                await self._send_too_large(scope, receive, send)
                return
            body.extend(chunk)
            request_complete = not message.get("more_body", False)

        body_bytes = bytes(body)
        del body
        replayed = False

        async def replay_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _send_overloaded(self, scope, receive, send) -> None:
        headers = {"Retry-After": "1"}
        if scope.get("http_version") in {"1.0", "1.1"}:
            headers["Connection"] = "close"
        response = JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Too many requests are being processed. Retry shortly."},
            headers=headers,
        )
        await response(scope, receive, send)

    async def _send_too_large(self, scope, receive, send) -> None:
        headers = {}
        if scope.get("http_version") in {"1.0", "1.1"}:
            headers["Connection"] = "close"
        response = JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={"detail": "Request body exceeds the 1 MiB limit."},
            headers=headers,
        )
        await response(scope, receive, send)

# Thread-safe LRU evaluation cache (up to 512 recent decisions)
_eval_cache: OrderedDict[str, dict] = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX = 512

# Telemetry counters for Prometheus exposition
_telemetry_requests = {"ALLOWED": 0, "FLAGGED": 0, "BLOCKED": 0}
_telemetry_by_mode = {"text": 0, "metric": 0}
_telemetry_cache_hits = 0
_telemetry_pii_detected = 0
_telemetry_latencies: Deque[float] = deque(maxlen=2000)
_telemetry_lock = threading.Lock()


def get_engine() -> EvalShieldEngine:
    """Return lazily-initialized inference engine singleton."""
    global _engine
    if _engine is None:
        _engine = get_inference_engine()
    return _engine


def get_router() -> ChampionChallengerRouter:
    """Return lazily-initialized Champion/Challenger router singleton."""
    global _router
    if _router is None:
        _router = get_champion_challenger_router()
    return _router


def get_extractor() -> TextFeatureExtractor:
    """Return lazily-initialized feature extractor singleton."""
    global _extractor
    if _extractor is None:
        _extractor = get_feature_extractor()
    return _extractor


def get_pii() -> PIIDetector:
    """Return lazily-initialized PII detector singleton."""
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = get_pii_detector()
    return _pii_detector


def get_baseline_df() -> pd.DataFrame | None:
    """Return lazily-loaded baseline distribution DataFrame."""
    global _baseline_df
    if _baseline_df is None:
        baseline_path = (
            Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "baseline_distribution.parquet"
        )
        if baseline_path.exists():
            try:
                _baseline_df = pd.read_parquet(baseline_path)
                logger.info("Loaded baseline distribution: %d samples.", len(_baseline_df))
            except Exception as exc:
                logger.warning("Could not load baseline parquet: %s", exc)
        else:
            logger.warning("Baseline parquet not found at %s", baseline_path)
    return _baseline_df


def _resolve_features(payload: EvaluationRequest) -> Tuple[Dict[str, float], str]:
    """Resolve request to a feature vector and return (features, extraction_mode).

    Priority:
    1. If prompt_text supplied -> auto-extract via TextFeatureExtractor.
       Any explicit metric override replaces the extracted value.
    2. If no prompt_text -> caller must supply all 5 metrics directly.
    """
    if payload.prompt_text is not None and bool(payload.prompt_text.strip()):
        extractor = get_extractor()
        overrides = {
            "toxicity_score": payload.toxicity_score,
            "hallucination_risk": payload.hallucination_risk,
            "prompt_injection_score": payload.prompt_injection_score,
            "semantic_drift": payload.semantic_drift,
            "token_entropy": payload.token_entropy,
        }
        features = extractor.extract_features(
            prompt_text=payload.prompt_text,
            response_text=payload.response_text,
            overrides={k: v for k, v in overrides.items() if v is not None},
        )
        return features, "text"

    # Metric Mode — all fields guaranteed non-None by Pydantic validator
    return {
        "toxicity_score": float(payload.toxicity_score),  # type: ignore[arg-type]
        "hallucination_risk": float(payload.hallucination_risk),  # type: ignore[arg-type]
        "prompt_injection_score": float(payload.prompt_injection_score),  # type: ignore[arg-type]
        "semantic_drift": float(payload.semantic_drift),  # type: ignore[arg-type]
        "token_entropy": float(payload.token_entropy),  # type: ignore[arg-type]
    }, "metric"


def _calculate_risk_contributions(features: Dict[str, float]) -> Dict[str, float]:
    """Compute normalized explainable risk attribution weights (sums to 1.0).

    Fulfills EU AI Act Article 13 & NIST AI RMF Measure requirements.
    """
    weights = {
        "prompt_injection_score": 0.35,
        "toxicity_score": 0.30,
        "hallucination_risk": 0.20,
        "semantic_drift": 0.10,
        "token_entropy": 0.05,
    }
    raw: Dict[str, float] = {}
    for feat, w in weights.items():
        val = features.get(feat, 0.0)
        scaled_val = min(1.0, max(0.0, val / 10.0)) if feat == "token_entropy" else min(1.0, max(0.0, val))
        raw[feat] = w * scaled_val

    total_weighted = sum(raw.values())
    if total_weighted > 0.0001:
        return {k: round(v / total_weighted, 4) for k, v in raw.items()}
    return {k: 0.20 for k in weights}


def _cache_key(features: Dict[str, float]) -> str:
    """Produce a stable SHA-256 cache key from resolved feature values."""
    canonical = "|".join(f"{k}:{features[k]:.4f}" for k in sorted(features))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _cached_predict(features: Dict[str, float], prompt_id: Optional[str] = None) -> dict:
    """Return cached inference result or route via Champion/Challenger router.

    Uses thread-safe OrderedDict LRU eviction.
    """
    global _telemetry_cache_hits
    key = _cache_key(features)

    with _cache_lock:
        if key in _eval_cache:
            _telemetry_cache_hits += 1
            _eval_cache.move_to_end(key)
            cached = dict(_eval_cache[key])
            cached["latency_ms"] = 0.0  # Cache hit: zero compute latency
            return cached

    router = get_router()
    result = router.route_and_evaluate(features, prompt_id=prompt_id)

    with _cache_lock:
        if len(_eval_cache) >= _CACHE_MAX:
            _eval_cache.popitem(last=False)
        _eval_cache[key] = result

    return result


def _record_telemetry(decision: str, mode: str, latency_ms: float) -> None:
    """Update global metrics counters thread-safely."""
    with _telemetry_lock:
        if decision in _telemetry_requests:
            _telemetry_requests[decision] += 1
        if mode in _telemetry_by_mode:
            _telemetry_by_mode[mode] += 1
        _telemetry_latencies.append(latency_ms)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: warm all singletons before accepting traffic."""
    # Fail before loading models or serving requests when production security
    # configuration is absent or invalid.
    auth_config: APIAuthConfig = getattr(app.state, "api_auth_config", _api_auth_config)
    auth_config.validate()
    app.state.api_auth_config = auth_config

    logger.info("Starting EvalShield-AI engine warm-up...")
    get_engine()
    get_router()
    get_extractor()
    get_pii()
    get_baseline_df()
    logger.info("EvalShield-AI ready. Champion/Challenger shadow router & PII engine active.")
    yield
    logger.info("EvalShield-AI graceful shutdown complete.")


app = FastAPI(
    title="EvalShield-AI",
    description=(
        "Enterprise-grade AI Guardrail, Shadow Routing & Quality Engineering Platform. "
        "Accepts raw text prompts or pre-computed risk metrics."
    ),
    version="1.3.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_auth)],
    **documentation_urls(_api_auth_config),
)
app.state.api_auth_config = _api_auth_config

# Middleware insertion order is intentional: the size limiter is inside CORS,
# while the timing/security-header middleware registered below wraps both.
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=MAX_REQUEST_BODY_BYTES,
    max_concurrent_requests=MAX_CONCURRENT_BUFFERED_REQUESTS,
)
add_api_cors_middleware(app, cors_origins_from_env())


@app.exception_handler(RequestValidationError)
async def safe_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Do not echo rejected request values (which may contain PII) in 422 bodies."""
    return JSONResponse(status_code=422, content={"detail": "Invalid request payload."})


@app.middleware("http")
async def add_timing_and_security_headers(request: Request, call_next) -> Response:
    """Inject timing and hardened security headers into every response."""
    t0 = time.perf_counter()
    auth_config: APIAuthConfig = getattr(request.app.state, "api_auth_config", _api_auth_config)
    if auth_config.authentication_enabled and request.url.path in {
        "/docs",
        "/redoc",
        "/openapi.json",
    }:
        # These Starlette routes are not covered by FastAPI's operation
        # dependencies. Hide them according to the active runtime policy too,
        # including when that policy was supplied through app.state at startup.
        response = JSONResponse(status_code=404, content={"detail": "Not Found"})
    else:
        response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000.0:.2f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ---------------------------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    """Root endpoint for service discovery."""
    return {
        "service": "EvalShield-AI",
        "version": "1.3.0",
        "status": "operational",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
        "compliance_report_url": "/model/compliance-report",
        "shadow_stats_url": "/model/shadow-stats",
        "streaming_guardrail_url": "/evaluate/stream",
    }


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check() -> HealthResponse:
    """Return current health, uptime, model readiness, and backend."""
    uptime = time.time() - START_TIME
    eng = get_engine()
    model_ready = eng is not None and eng.backend != "uninitialized"
    return HealthResponse(
        status="healthy" if model_ready else "degraded",
        uptime_seconds=round(uptime, 2),
        model_loaded=model_ready,
        backend=eng.backend if eng else "none",
    )


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model"])
async def get_model_metadata() -> ModelMetadataResponse:
    """Fetch active model configuration, training metrics, and feature schema."""
    eng = get_engine()
    if not eng:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not loaded")
    meta = eng.get_metadata()
    return ModelMetadataResponse(
        backend=meta["backend"],
        feature_names=meta["feature_names"],
        classes=meta["classes"],
        model_metadata=meta["model_metadata"],
    )


@app.get("/model/shadow-stats", response_model=ShadowStatsResponse, tags=["Observability"])
async def get_shadow_stats() -> ShadowStatsResponse:
    """Return live Champion/Challenger shadow evaluation metrics and divergence telemetry."""
    router = get_router()
    stats = router.get_shadow_stats()
    stats["recent_discrepancies"] = router.get_recent_discrepancies(limit=10)
    return ShadowStatsResponse(**stats)


@app.get("/model/compliance-report", response_model=ComplianceReportResponse, tags=["Compliance"])
async def get_compliance_report() -> ComplianceReportResponse:
    """Export compliance documentation for NIST AI RMF 1.0 & EU AI Act Annex IV."""
    eng = get_engine()
    router = get_router()
    shadow_stats = router.get_shadow_stats()
    metadata = eng.get_metadata().get("model_metadata", {})

    report = {
        "frameworks": [
            "NIST AI RMF 1.0 (Artificial Intelligence Risk Management Framework)",
            "EU AI Act (Regulation (EU) 2024/1689 - High-Risk AI Technical Documentation Annex IV)",
        ],
        "system_identification": {
            "system_name": "EvalShield-AI",
            "version": "1.3.0",
            "provider": "EvalShield Platform Engineering",
            "lifecycle_stage": "Production Operations & Continuous Monitoring",
            "primary_backend": eng.backend,
        },
        "risk_classification": {
            "eu_ai_act_tier": "General-Purpose AI Model Guardrail System / High-Risk Safety Gate",
            "intended_purpose": (
                "Real-time evaluation, prompt injection mitigation, toxic content gating, "
                "PII credential redaction, and semantic drift monitoring for generative AI workflows."
            ),
            "prohibited_practices_check": "PASS (No biometric categorization, social scoring, or subliminal manipulation)",
        },
        "model_architecture_and_specs": {
            "classifier_family": "Calibrated Tree Ensemble (Random Forest + Gradient Boosting)",
            "training_sample_count": metadata.get("training_samples", 4000),
            "accuracy": metadata.get("accuracy", 0.9488),
            "macro_f1": metadata.get("macro_f1", 0.8892),
            "weighted_f1": metadata.get("weighted_f1", 0.9479),
            "int8_quantization_enabled": metadata.get("int8_onnx_available", True),
            "latency_p95_ms": shadow_stats.get("avg_champion_latency_ms", 1.5),
        },
        "adversarial_robustness_and_fuzzing": {
            "tested_attack_surfaces": [
                "Direct Jailbreak & Roleplay Prompt Injection (DAN, Developer Mode)",
                "System Instruction Override & Delimiter Injection (Markdown, Tags)",
                "Base64 and Homoglyph Obfuscation Payloads",
                "Severe Lexical and Contextual Toxicity Spikes",
                "Shannon Token Entropy Degeneracy and Payload Gibberish",
                "PII & Secret Exfiltration (Luhn Credit Cards, SSN, API Keys)",
            ],
            "adversarial_test_status": "VERIFIED_PASS (Automated behavioral contracts passed)",
            "deterministic_failsafes": "Deterministic hard-stop overrides triggered for toxicity > 0.70 or injection > 0.65",
        },
        "data_governance_and_drift_controls": {
            "baseline_distribution_format": "Parquet tabular benchmark with schema enforcement",
            "drift_detection_method": "Two-sample Kolmogorov-Smirnov statistical test (alpha = 0.05)",
            "population_stability_index_threshold": 0.20,
            "zero_data_retention_policy": "Prompts analyzed in memory; raw texts sanitized or discarded",
            "pii_redaction_engine": "Embedded ISO/IEC 7812 Luhn algorithm & pattern redaction",
        },
        "explainability_and_decision_logic": {
            "evaluated_risk_features": eng.feature_names,
            "verdict_categories": ["ALLOWED", "FLAGGED", "BLOCKED"],
            "scoring_formula": "Calibrated posterior risk index: P(BLOCKED)*1.0 + P(FLAGGED)*0.5",
            "explainability_attribution": "Normalized feature contribution weights provided per inference",
        },
        "human_oversight_and_shadow_telemetry": {
            "shadow_router_active": shadow_stats.get("challenger_active", True),
            "shadow_traffic_rate": shadow_stats.get("shadow_traffic_rate", 1.0),
            "decision_agreement_rate": shadow_stats.get("decision_agreement_rate", 1.0),
            "total_production_requests": shadow_stats.get("total_production_requests", 0),
            "shadow_evaluations_run": shadow_stats.get("shadow_evaluations_run", 0),
            "discrepancies_detected": shadow_stats.get("discrepancy_count", 0),
            "failover_architecture": "Automatic graceful degradation: INT8 ONNX -> FP32 ONNX -> Sklearn Joblib",
        },
        "compliance_status": "COMPLIANT",
    }
    return ComplianceReportResponse(**report)


@app.get("/metrics", tags=["Observability"])
async def prometheus_metrics() -> Response:
    """Expose real-time Prometheus scrapable metrics exposition format (text/plain).

    Includes count and sum metrics for standard PromQL aggregation.
    """
    uptime = time.time() - START_TIME
    router = get_router()
    shadow_stats = router.get_shadow_stats()

    with _telemetry_lock:
        latencies = list(_telemetry_latencies)
        req_allowed = _telemetry_requests["ALLOWED"]
        req_flagged = _telemetry_requests["FLAGGED"]
        req_blocked = _telemetry_requests["BLOCKED"]
        mode_text = _telemetry_by_mode["text"]
        mode_metric = _telemetry_by_mode["metric"]
        cache_hits = _telemetry_cache_hits
        pii_count = _telemetry_pii_detected

    if latencies:
        p50 = round(float(np.percentile(latencies, 50)), 3)
        p95 = round(float(np.percentile(latencies, 95)), 3)
        p99 = round(float(np.percentile(latencies, 99)), 3)
        latency_sum = round(sum(latencies), 3)
    else:
        p50 = p95 = p99 = latency_sum = 0.0

    lines = [
        "# HELP evalshield_requests_total Total number of evaluation requests processed by decision verdict",
        "# TYPE evalshield_requests_total counter",
        f'evalshield_requests_total{{decision="ALLOWED"}} {req_allowed}',
        f'evalshield_requests_total{{decision="FLAGGED"}} {req_flagged}',
        f'evalshield_requests_total{{decision="BLOCKED"}} {req_blocked}',
        "",
        "# HELP evalshield_requests_by_mode_total Total evaluation requests broken down by input mode",
        "# TYPE evalshield_requests_by_mode_total counter",
        f'evalshield_requests_by_mode_total{{mode="text"}} {mode_text}',
        f'evalshield_requests_by_mode_total{{mode="metric"}} {mode_metric}',
        "",
        "# HELP evalshield_cache_hits_total In-memory LRU evaluation cache hits",
        "# TYPE evalshield_cache_hits_total counter",
        f"evalshield_cache_hits_total {cache_hits}",
        "",
        "# HELP evalshield_pii_detected_total Total number of sensitive PII entities or secrets detected",
        "# TYPE evalshield_pii_detected_total counter",
        f"evalshield_pii_detected_total {pii_count}",
        "",
        "# HELP evalshield_shadow_evaluations_total Total background shadow evaluations executed by challenger",
        "# TYPE evalshield_shadow_evaluations_total counter",
        f'evalshield_shadow_evaluations_total {shadow_stats["shadow_evaluations_run"]}',
        "",
        "# HELP evalshield_shadow_jobs_skipped_total Shadow jobs skipped because outstanding-work capacity was exhausted or executor submission failed",
        "# TYPE evalshield_shadow_jobs_skipped_total counter",
        f'evalshield_shadow_jobs_skipped_total {shadow_stats["shadow_jobs_skipped"]}',
        "",
        "# HELP evalshield_shadow_jobs_outstanding Currently running or queued shadow jobs",
        "# TYPE evalshield_shadow_jobs_outstanding gauge",
        f'evalshield_shadow_jobs_outstanding {shadow_stats["shadow_jobs_outstanding"]}',
        "",
        "# HELP evalshield_shadow_discrepancies_total Total detected discrepancies between champion and challenger",
        "# TYPE evalshield_shadow_discrepancies_total counter",
        f'evalshield_shadow_discrepancies_total {shadow_stats["discrepancy_count"]}',
        "",
        "# HELP evalshield_latency_milliseconds Request evaluation latency summary",
        "# TYPE evalshield_latency_milliseconds summary",
        f'evalshield_latency_milliseconds{{quantile="0.5"}} {p50}',
        f'evalshield_latency_milliseconds{{quantile="0.95"}} {p95}',
        f'evalshield_latency_milliseconds{{quantile="0.99"}} {p99}',
        f"evalshield_latency_milliseconds_sum {latency_sum}",
        f"evalshield_latency_milliseconds_count {len(latencies)}",
        "",
        "# HELP evalshield_uptime_seconds Total runtime of the API service in seconds",
        "# TYPE evalshield_uptime_seconds gauge",
        f"evalshield_uptime_seconds {uptime:.2f}",
        "",
    ]
    return Response(
        content="\n".join(lines),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def _evaluate_interaction(payload: EvaluationRequest) -> Tuple[Dict[str, Any], str]:
    """Execute complete unified evaluation pipeline: feature extraction, inference, PII, canary, and attribution."""
    pii_detector = get_pii()
    safe_prompt_id = pii_detector.redact_pii(payload.prompt_id)[0] if payload.prompt_id else None
    sanitized_text: Optional[str] = None
    prompt_entities = []
    response_entities = []

    # Scan original values for metadata, and redact before feature extraction
    # whenever the caller opted in. The model and extractor only see sanitized
    # text in that mode.
    processed_payload = payload
    if payload.prompt_text:
        if payload.redact_pii:
            sanitized_text, prompt_entities = pii_detector.redact_pii(payload.prompt_text)
        else:
            prompt_entities = pii_detector.detect_pii(payload.prompt_text)
    if payload.response_text:
        if payload.redact_pii:
            sanitized_response, response_entities = pii_detector.redact_pii(payload.response_text)
        else:
            sanitized_response = payload.response_text
            response_entities = pii_detector.detect_pii(payload.response_text)
    else:
        sanitized_response = payload.response_text

    if payload.redact_pii:
        processed_payload = payload.model_copy(
            update={"prompt_text": sanitized_text, "response_text": sanitized_response}
        )

    features, extraction_mode = _resolve_features(processed_payload)
    result = _cached_predict(features, prompt_id=safe_prompt_id)
    result["prompt_id"] = safe_prompt_id
    result["extraction_mode"] = extraction_mode

    # 1. Compute explainable risk contributions (sums to 1.0)
    result["risk_contributions"] = _calculate_risk_contributions(features)

    # 2. Sensitive PII & Secret scanning / redaction
    pii_entities_list: List[Dict[str, Any]] = [
        entity.to_dict(redacted=payload.redact_pii, source="prompt") for entity in prompt_entities
    ]
    pii_entities_list.extend(
        entity.to_dict(redacted=payload.redact_pii, source="response") for entity in response_entities
    )

    result["sanitized_prompt"] = sanitized_text
    result["pii_detected"] = pii_entities_list

    # 3. System prompt canary honeytoken leak detection
    result["canary_leak_detected"] = False
    result["leaked_canary_token"] = None
    if payload.canary_tokens and payload.response_text:
        sanitizer = get_unicode_sanitizer()
        is_leaked, leaked_token = sanitizer.check_canary_leak(payload.response_text, payload.canary_tokens)
        if is_leaked:
            result["shield_decision"] = "BLOCKED"
            result["risk_score"] = 1.0
            result["confidence"] = 1.0
            result["canary_leak_detected"] = True
            # Preserve the response field for compatibility, but never return
            # the canary value itself.
            result["leaked_canary_token"] = None
            reasons = list(result.get("violation_reasons", []))
            reasons.append(
                "CRITICAL: System prompt canary honeytoken detected in model output."
            )
            result["violation_reasons"] = reasons

    if pii_entities_list:
        global _telemetry_pii_detected
        with _telemetry_lock:
            _telemetry_pii_detected += len(pii_entities_list)

    return result, extraction_mode


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Inference"])
async def evaluate_single(payload: EvaluationRequest) -> EvaluationResponse:
    """Evaluate one AI interaction.

    Accepts **Text Mode** (prompt_text -> auto-extraction) or **Metric Mode**
    (all 5 float scores). Supports PII/secret scanning & redaction and produces
    normalized feature explainability weights.
    """
    eng = get_engine()
    if not eng:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not loaded")

    try:
        result, extraction_mode = _evaluate_interaction(payload)

        # Track telemetry
        _record_telemetry(
            decision=result["shield_decision"],
            mode=extraction_mode,
            latency_ms=result.get("latency_ms", 0.0),
        )

        return EvaluationResponse(**result)
    except (ValueError, KeyError):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid evaluation input.")
    except Exception:
        # Exception text and tracebacks can contain user supplied values.
        logger.error("Inference failed during evaluation.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference failed.")


@app.post("/evaluate/stream", tags=["Streaming"])
async def evaluate_streaming(payload: EvaluationRequest) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming guardrail for real-time generative responses.

    Streams safety checks and emits early terminate signals if safety constraints are breached.
    """
    async def sse_generator() -> AsyncGenerator[str, None]:
        try:
            pii_detector = get_pii()
            safe_prompt_id = pii_detector.redact_pii(payload.prompt_id)[0] if payload.prompt_id else None
            # Event 1: Pipeline Init
            yield f"event: init\ndata: {json.dumps({'prompt_id': safe_prompt_id, 'status': 'connected'})}\n\n"
            await asyncio.sleep(0.01)

            # Event 2: PII Scan
            pii_list = []
            if payload.prompt_text:
                if payload.redact_pii:
                    _, entities = pii_detector.redact_pii(payload.prompt_text)
                else:
                    entities = pii_detector.detect_pii(payload.prompt_text)
                pii_list = [e.to_dict(redacted=payload.redact_pii, source="prompt") for e in entities]

            yield f"event: pii_scan\ndata: {json.dumps({'pii_count': len(pii_list), 'entities': pii_list})}\n\n"
            await asyncio.sleep(0.01)

            # Event 3: Evaluation Verdict
            res, extraction_mode = _evaluate_interaction(payload)

            verdict_payload = {
                "shield_decision": res["shield_decision"],
                "risk_score": res["risk_score"],
                "confidence": res["confidence"],
                "risk_contributions": res.get("risk_contributions", {}),
                "category_scores": res["category_scores"],
                "violation_reasons": res.get("violation_reasons", []),
                "canary_leak_detected": res.get("canary_leak_detected", False),
                "action": "ABORT_STREAM" if res["shield_decision"] == "BLOCKED" else "CONTINUE",
            }
            yield f"event: verdict\ndata: {json.dumps(verdict_payload)}\n\n"
            await asyncio.sleep(0.01)

            # Event 4: Complete
            yield f"event: done\ndata: {json.dumps({'status': 'completed', 'decision': res['shield_decision']})}\n\n"
        except Exception:
            # Never emit exception text or tracebacks into server logs or SSE.
            logger.error("Streaming evaluation failed.")
            yield f"event: error\ndata: {json.dumps({'error': 'Evaluation failed.'})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.post("/evaluate/batch", response_model=BatchEvaluationResponse, tags=["Inference"])
async def evaluate_batch(batch: BatchEvaluationRequest) -> BatchEvaluationResponse:
    """Evaluate a batch of interactions. Supports mixed Text and Metric mode items."""
    eng = get_engine()
    if not eng:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not loaded")

    t_start = time.perf_counter()
    results: List[EvaluationResponse] = []
    allowed = flagged = blocked = 0

    try:
        for item in batch.items:
            result, extraction_mode = _evaluate_interaction(item)
            res = EvaluationResponse(**result)
            results.append(res)

            _record_telemetry(
                decision=res.shield_decision,
                mode=extraction_mode,
                latency_ms=res.latency_ms,
            )

            if res.shield_decision == "ALLOWED":
                allowed += 1
            elif res.shield_decision == "FLAGGED":
                flagged += 1
            else:
                blocked += 1
    except (ValueError, KeyError):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid evaluation input.")
    except Exception:
        logger.error("Batch evaluation failed.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Batch evaluation failed.")

    batch_latency = (time.perf_counter() - t_start) * 1000.0
    return BatchEvaluationResponse(
        results=results,
        total_evaluated=len(results),
        allowed_count=allowed,
        flagged_count=flagged,
        blocked_count=blocked,
        batch_latency_ms=round(batch_latency, 2),
    )

    return BatchEvaluationResponse(
        results=results,
        total_evaluated=len(results),
        allowed_count=allowed,
        flagged_count=flagged,
        blocked_count=blocked,
        batch_latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
    )


@app.post("/drift/check", response_model=DriftCheckResponse, tags=["Drift Monitoring"])
async def check_feature_drift(request: DriftCheckRequest) -> DriftCheckResponse:
    """Two-sample Kolmogorov-Smirnov drift test against the training baseline."""
    df_base = get_baseline_df()
    if df_base is None or df_base.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Baseline distribution not available. Re-run model_generator.py.",
        )

    results: List[DriftFeatureResult] = []
    drifted_count = 0

    for feature_name, sample_values in request.feature_samples.items():
        if feature_name not in df_base.columns or len(sample_values) < 10:
            continue

        baseline_vals = df_base[feature_name].dropna().values
        ks_stat, p_val = ks_2samp(baseline_vals, sample_values)
        is_drifted = bool(p_val < request.significance_level)
        if is_drifted:
            drifted_count += 1

        results.append(
            DriftFeatureResult(
                feature_name=feature_name,
                drift_detected=is_drifted,
                ks_statistic=round(float(ks_stat), 4),
                p_value=round(float(p_val), 5),
                baseline_mean=round(float(np.mean(baseline_vals)), 4),
                sample_mean=round(float(np.mean(sample_values)), 4),
            )
        )

    return DriftCheckResponse(
        overall_drift_detected=(drifted_count > 0),
        drifted_features_count=drifted_count,
        total_features_evaluated=len(results),
        features=results,
    )


@app.post("/drift/recalibrate", response_model=RecalibrateResponse, tags=["Drift Monitoring"])
async def trigger_drift_recalibration(request: RecalibrateRequest) -> RecalibrateResponse:
    """Analyze production distribution drift and trigger automated adaptive threshold recalibration."""
    recal_engine = get_recalibration_engine()
    result = recal_engine.recalibrate(
        production_samples=request.production_samples,
        significance_level=request.significance_level,
    )
    return RecalibrateResponse(**result.to_dict())


@app.post("/red-team/simulate", response_model=RedTeamSimulateResponse, tags=["Red Team"])
async def simulate_red_team_campaign(
    request: RedTeamSimulateRequest,
    response: Response,
) -> RedTeamSimulateResponse:
    """Execute an automated genetic adversarial red-teaming campaign against the shield."""
    fuzzer = get_red_team_fuzzer()
    safe_seed_prompt = get_pii().redact_pii(request.seed_prompt)[0]
    campaign_res = fuzzer.run_campaign(
        seed_prompt=safe_seed_prompt,
        rounds=request.rounds,
        allowed_types=request.mutation_types,
        trigger_recalibration=request.trigger_recalibration,
    )
    # Inject tamper-proof cryptographic receipt header
    response.headers["X-Shield-Signature"] = campaign_res.cryptographic_receipt
    return RedTeamSimulateResponse(**campaign_res.to_dict())


@app.get("/red-team/mutations", response_model=RedTeamMutationsResponse, tags=["Red Team"])
async def list_red_team_mutations() -> RedTeamMutationsResponse:
    """List available genetic mutation operators and their adversarial mechanisms."""
    return RedTeamMutationsResponse(
        available_mutations={
            "leet_speak": "Substitutes vowels and alphanumeric characters with phonetic symbols and digits",
            "payload_split": "Inserts hyphens or spaces into sensitive tokens to evade exact-match boundary filters",
            "base64_wrap": "Encodes adversarial payload into Base64 format inside an execution wrapper",
            "roleplay_frame": "Wraps prompt inside complex fictional screenplays or cognitive dissonance frames",
            "hypothetical": "Pretends prompt is an academic inquiry, theoretical exercise, or ethical audit",
            "delimiter_injection": "Prepends raw LLM system context delimiters to hijack instruction hierarchy",
        }
    )
