"""EvalShield-AI: Production FastAPI Application.

High-throughput, observable inference & evaluation API with built-in
drift testing, batch processing, and health telemetry.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import ks_2samp

from src.api.schemas import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    DriftCheckRequest,
    DriftCheckResponse,
    DriftFeatureResult,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    ModelMetadataResponse,
)
from src.ml.inference import EvalShieldEngine, get_inference_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evalshield_api")

START_TIME = time.time()
engine: EvalShieldEngine | None = None
baseline_df: pd.DataFrame | None = None


def get_engine() -> EvalShieldEngine:
    """Ensure inference engine singleton is loaded."""
    global engine
    if engine is None:
        engine = get_inference_engine()
    return engine


def get_baseline_df() -> pd.DataFrame | None:
    """Ensure baseline distribution parquet is loaded."""
    global baseline_df
    if baseline_df is None:
        baseline_path = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "baseline_distribution.parquet"
        if baseline_path.exists():
            try:
                baseline_df = pd.read_parquet(baseline_path)
                logger.info("Loaded baseline distribution containing %d samples.", len(baseline_df))
            except Exception as e:
                logger.warning("Could not read baseline distribution parquet: %s", e)
        else:
            logger.warning("Baseline distribution parquet not found at %s", baseline_path)
    return baseline_df


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle management."""
    logger.info("Initializing EvalShield-AI Inference Engine...")
    eng = get_engine()
    get_baseline_df()
    logger.info("EvalShield-AI API ready for traffic on backend: %s", eng.backend)
    yield
    logger.info("Shutting down EvalShield-AI...")


app = FastAPI(
    title="EvalShield-AI Engine",
    description="Enterprise-grade AI Guardrail, Evaluation & Drift Detection Platform.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next) -> Response:
    """Track request processing latency and inject into response headers."""
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    """Root endpoint for discovery."""
    return {
        "service": "EvalShield-AI",
        "status": "operational",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check() -> HealthResponse:
    """Return system health, uptime, and model loading state."""
    uptime = time.time() - START_TIME
    active_engine = get_engine()
    model_ready = active_engine is not None and active_engine.backend != "uninitialized"
    status_str = "healthy" if model_ready else "degraded"

    return HealthResponse(
        status=status_str,
        uptime_seconds=round(uptime, 2),
        model_loaded=model_ready,
        backend=active_engine.backend if active_engine else "none",
    )


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model"])
async def get_model_metadata() -> ModelMetadataResponse:
    """Fetch active model configuration, training parameters, and performance."""
    active_engine = get_engine()
    if not active_engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not loaded")
    meta = active_engine.get_metadata()
    return ModelMetadataResponse(
        backend=meta["backend"],
        feature_names=meta["feature_names"],
        classes=meta["classes"],
        model_metadata=meta["model_metadata"],
    )


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Inference"])
async def evaluate_single(payload: EvaluationRequest) -> EvaluationResponse:
    """Evaluate a single AI interaction across risk guardrails."""
    active_engine = get_engine()
    if not active_engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not loaded")

    features = {
        "toxicity_score": payload.toxicity_score,
        "hallucination_risk": payload.hallucination_risk,
        "prompt_injection_score": payload.prompt_injection_score,
        "semantic_drift": payload.semantic_drift,
        "token_entropy": payload.token_entropy,
    }

    try:
        result = active_engine.predict(features)
        result["prompt_id"] = payload.prompt_id
        return EvaluationResponse(**result)
    except Exception as e:
        logger.error("Inference failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")


@app.post("/evaluate/batch", response_model=BatchEvaluationResponse, tags=["Inference"])
async def evaluate_batch(batch: BatchEvaluationRequest) -> BatchEvaluationResponse:
    """Evaluate a batch of AI interactions concurrently."""
    active_engine = get_engine()
    if not active_engine:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not loaded")

    t_start = time.perf_counter()
    results: List[EvaluationResponse] = []
    allowed = 0
    flagged = 0
    blocked = 0

    for item in batch.items:
        features = {
            "toxicity_score": item.toxicity_score,
            "hallucination_risk": item.hallucination_risk,
            "prompt_injection_score": item.prompt_injection_score,
            "semantic_drift": item.semantic_drift,
            "token_entropy": item.token_entropy,
        }
        pred = active_engine.predict(features)
        pred["prompt_id"] = item.prompt_id
        res = EvaluationResponse(**pred)
        results.append(res)

        if res.shield_decision == "ALLOWED":
            allowed += 1
        elif res.shield_decision == "FLAGGED":
            flagged += 1
        else:
            blocked += 1

    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    return BatchEvaluationResponse(
        results=results,
        total_evaluated=len(results),
        allowed_count=allowed,
        flagged_count=flagged,
        blocked_count=blocked,
        batch_latency_ms=round(total_time_ms, 2),
    )


@app.post("/drift/check", response_model=DriftCheckResponse, tags=["Drift Monitoring"])
async def check_feature_drift(request: DriftCheckRequest) -> DriftCheckResponse:
    """Compare sample distributions with baseline distributions using Kolmogorov-Smirnov test."""
    df_base = get_baseline_df()
    if df_base is None or df_base.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Baseline distribution data not available. Retrain model to generate baseline.",
        )

    results: List[DriftFeatureResult] = []
    drifted_count = 0

    for feature_name, sample_values in request.feature_samples.items():
        if feature_name not in df_base.columns:
            continue
        if len(sample_values) < 10:
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
