"""EvalShield-AI: Model Generator and Training Pipeline.

Generates realistic AI safety benchmarks, trains a risk classifier,
evaluates performance metrics, logs metadata with MLflow, and exports
production artifacts to both Joblib and ONNX formats.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# MLflow logging
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ONNX export
try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("model_generator")

FEATURE_NAMES = [
    "toxicity_score",
    "hallucination_risk",
    "prompt_injection_score",
    "semantic_drift",
    "token_entropy",
]

CLASS_NAMES = ["ALLOWED", "FLAGGED", "BLOCKED"]
CLASS_MAPPING = {0: "ALLOWED", 1: "FLAGGED", 2: "BLOCKED"}


def generate_synthetic_benchmark(
    n_samples: int = 4000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate deterministic, statistically grounded AI evaluation benchmark data."""
    rng = np.random.RandomState(random_state)

    # 1. Toxicity score (beta distribution skewed towards low values with rare spikes)
    toxicity = rng.beta(0.5, 4.0, size=n_samples)

    # 2. Hallucination risk (normal distribution clipped between 0.0 and 1.0)
    hallucination = np.clip(rng.normal(loc=0.25, scale=0.2, size=n_samples), 0.0, 1.0)

    # 3. Prompt injection score (exponentially distributed)
    prompt_injection = np.clip(rng.exponential(scale=0.15, size=n_samples), 0.0, 1.0)

    # 4. Semantic drift (uniform distribution with subtle drift variations)
    drift = rng.uniform(0.0, 1.0, size=n_samples)

    # 5. Token entropy (normal distribution around 4.5, standard dev 1.8, bounded [0, 10])
    token_entropy = np.clip(rng.normal(loc=4.5, scale=1.8, size=n_samples), 0.0, 10.0)

    df = pd.DataFrame({
        "toxicity_score": toxicity,
        "hallucination_risk": hallucination,
        "prompt_injection_score": prompt_injection,
        "semantic_drift": drift,
        "token_entropy": token_entropy,
    })

    # Realistic Ground Truth Decision Rules
    # Base risk calculation:
    risk_factor = (
        0.35 * df["prompt_injection_score"]
        + 0.30 * df["toxicity_score"]
        + 0.20 * df["hallucination_risk"]
        + 0.10 * df["semantic_drift"]
        + 0.05 * (df["token_entropy"] / 10.0)
    )

    # Class assignment:
    # 0 = ALLOWED: risk < 0.28 AND prompt_injection < 0.35 AND toxicity < 0.40
    # 1 = FLAGGED: 0.28 <= risk < 0.55
    # 2 = BLOCKED: risk >= 0.55 OR prompt_injection >= 0.70 OR toxicity >= 0.75
    labels = np.zeros(n_samples, dtype=int)
    flagged_mask = (risk_factor >= 0.25) & (risk_factor < 0.55)
    blocked_mask = (risk_factor >= 0.55) | (df["prompt_injection_score"] >= 0.70) | (df["toxicity_score"] >= 0.75)

    labels[flagged_mask] = 1
    labels[blocked_mask] = 2

    # Add realistic 2% noise to simulate labeling ambiguity
    noise_mask = rng.rand(n_samples) < 0.02
    random_shifts = rng.choice([-1, 1], size=n_samples)
    labels[noise_mask] = np.clip(labels[noise_mask] + random_shifts[noise_mask], 0, 2)

    return df, pd.Series(labels, name="shield_decision")


def train_evalshield_pipeline(
    df: pd.DataFrame,
    labels: pd.Series,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train Gradient Boosting pipeline with scaling and evaluate metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.20, random_state=random_state, stratify=labels
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                random_state=random_state,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "classification_report": report,
        "test_sample_size": len(y_test),
        "train_sample_size": len(X_train),
    }

    logger.info(
        "Model evaluation: Accuracy=%.4f | Macro F1=%.4f | Weighted F1=%.4f",
        acc,
        f1_macro,
        f1_weighted,
    )
    return pipeline, metrics


def export_to_onnx(pipeline: Pipeline, output_path: Path) -> bool:
    """Export trained sklearn pipeline to ONNX format."""
    if not ONNX_AVAILABLE:
        logger.warning("skl2onnx/onnx not installed. Skipping ONNX export.")
        return False

    try:
        initial_type = [("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))]
        try:
            onnx_model = convert_sklearn(
                pipeline,
                initial_types=initial_type,
                target_opset=15,
                options={"zipmap": False},
            )
        except Exception:
            onnx_model = convert_sklearn(
                pipeline,
                initial_types=initial_type,
                target_opset=15,
            )
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info("Exported ONNX model successfully to %s", output_path)
        return True
    except Exception as e:
        logger.warning("Failed to export ONNX model (%s). Will rely on Joblib fallback.", e)
        return False


def generate_shield_model(
    output_dir: str | Path | None = None,
    n_samples: int = 4000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Complete generator pipeline: synthesis, training, MLflow logging, export."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "artifacts"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d synthetic benchmark samples...", n_samples)
    df, labels = generate_synthetic_benchmark(n_samples=n_samples, random_state=random_state)

    # Save baseline distribution for drift tracking
    baseline_path = output_dir / "baseline_distribution.parquet"
    df.assign(target=labels).to_parquet(baseline_path, index=False)
    logger.info("Saved baseline distribution to %s", baseline_path)

    # MLflow tracking
    run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", str(output_dir / "mlruns")))
            mlflow.set_experiment("EvalShield-AI-Core")
            run = mlflow.start_run(run_name=f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        except Exception as e:
            logger.warning("MLflow tracking initialization failed (%s). Continuing without MLflow logging.", e)
            run = None

    pipeline, metrics = train_evalshield_pipeline(df, labels, random_state=random_state)

    # Save Joblib Pipeline
    joblib_path = output_dir / "evalshield_model.joblib"
    joblib.dump(pipeline, joblib_path)
    logger.info("Saved Joblib pipeline to %s", joblib_path)

    # Save ONNX Model
    onnx_path = output_dir / "evalshield_model.onnx"
    onnx_exported = export_to_onnx(pipeline, onnx_path)

    # Build Metadata
    metadata = {
        "model_name": "EvalShield-Guardrail-Classifier",
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_NAMES,
        "classes": CLASS_NAMES,
        "metrics": metrics,
        "onnx_available": onnx_exported,
        "thresholds": {
            "allowed_max_risk": 0.28,
            "blocked_min_risk": 0.55,
            "toxicity_limit": 0.70,
            "injection_limit": 0.65,
        },
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if MLFLOW_AVAILABLE and run:
        try:
            mlflow.log_params({
                "n_estimators": 100,
                "learning_rate": 0.08,
                "max_depth": 4,
                "n_samples": n_samples,
            })
            mlflow.log_metrics({
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            })
            mlflow.end_run()
        except Exception as e:
            logger.warning("MLflow logging failed (%s).", e)

    logger.info("Model generation pipeline finished successfully.")
    return metadata


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    generate_shield_model(output_dir=target_dir)
