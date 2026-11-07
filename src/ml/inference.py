"""EvalShield-AI: Inference Engine.

Loads ONNX Runtime or Scikit-learn model, executes high-throughput inference,
applies guardrail heuristics, and produces detailed risk assessments.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from .model_generator import CLASS_MAPPING, CLASS_NAMES, FEATURE_NAMES, generate_shield_model

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

logger = logging.getLogger("evalshield_inference")


class EvalShieldEngine:
    """Production Inference Engine with ONNX Runtime & Sklearn fallback."""

    def __init__(self, artifacts_dir: Optional[Path | str] = None) -> None:
        if artifacts_dir is None:
            self.artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        else:
            self.artifacts_dir = Path(artifacts_dir)

        self.onnx_path = self.artifacts_dir / "evalshield_model.onnx"
        self.joblib_path = self.artifacts_dir / "evalshield_model.joblib"
        self.metadata_path = self.artifacts_dir / "metadata.json"

        self.ort_session: Optional[Any] = None
        self.joblib_model: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names = FEATURE_NAMES
        self.backend = "uninitialized"

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Ensure artifacts exist, generate if missing, and load into memory."""
        if not self.joblib_path.exists() and not self.onnx_path.exists():
            logger.info("Artifacts not found in %s. Triggering model generation...", self.artifacts_dir)
            generate_shield_model(output_dir=self.artifacts_dir)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # 1. Attempt ONNX loading
        if ORT_AVAILABLE and self.onnx_path.exists():
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = int(os.getenv("ORT_NUM_THREADS", "2"))
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(
                    str(self.onnx_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self.backend = "onnx"
                logger.info("EvalShield loaded ONNX Runtime backend successfully.")
                return
            except Exception as e:
                logger.warning("Failed to initialize ONNX Runtime session: %s. Falling back to Joblib.", e)

        # 2. Sklearn/Joblib fallback
        if self.joblib_path.exists():
            self.joblib_model = joblib.load(self.joblib_path)
            self.backend = "sklearn_joblib"
            logger.info("EvalShield loaded Joblib pipeline backend.")
        else:
            raise RuntimeError(f"Could not load any model from {self.artifacts_dir}")

    def _extract_feature_vector(self, data: Dict[str, float]) -> np.ndarray:
        """Convert input dict to float32 numpy vector matching training features."""
        vector = []
        for feat in self.feature_names:
            if feat not in data:
                raise ValueError(f"Missing required feature: '{feat}'")
            val = float(data[feat])
            vector.append(val)
        return np.array(vector, dtype=np.float32)

    def _evaluate_heuristic_violations(self, features: Dict[str, float]) -> List[str]:
        """Apply deterministic safety constraints and boundary policies."""
        violations: List[str] = []
        toxicity_threshold = 0.70
        injection_threshold = 0.65
        entropy_lower = 0.5
        entropy_upper = 9.5

        if features["toxicity_score"] > toxicity_threshold:
            violations.append(
                f"Toxicity score ({features['toxicity_score']:.2f}) exceeds threshold ({toxicity_threshold:.2f})"
            )
        if features["prompt_injection_score"] > injection_threshold:
            violations.append(
                f"Prompt injection likelihood ({features['prompt_injection_score']:.2f}) "
                f"exceeds safety limit ({injection_threshold:.2f})"
            )
        if features["token_entropy"] < entropy_lower:
            violations.append(
                f"Abnormally low token entropy ({features['token_entropy']:.2f}) indicates repetitive or degenerate output"
            )
        elif features["token_entropy"] > entropy_upper:
            violations.append(
                f"Extreme token entropy ({features['token_entropy']:.2f}) indicates potential gibberish or payload obfuscation"
            )

        return violations

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Run single inference and return comprehensive risk assessment."""
        t_start = time.perf_counter()
        vec = self._extract_feature_vector(features).reshape(1, -1)
        violations = self._evaluate_heuristic_violations(features)

        if self.backend == "onnx" and self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: vec})
            raw_pred = int(outputs[0][0])
            # Probability tensor
            probs = outputs[1][0]
            if isinstance(probs, dict):
                prob_dict = {CLASS_NAMES[k]: float(v) for k, v in probs.items()}
            else:
                prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        else:
            raw_pred = int(self.joblib_model.predict(vec)[0])
            probs = self.joblib_model.predict_proba(vec)[0]
            prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        decision = CLASS_MAPPING.get(raw_pred, "FLAGGED")

        # Hard guardrail override: severe heuristic violations force BLOCKED
        if any("exceeds threshold" in v or "exceeds safety limit" in v for v in violations):
            decision = "BLOCKED"

        # Calculate composite risk score
        risk_score = float(prob_dict.get("BLOCKED", 0.0) * 1.0 + prob_dict.get("FLAGGED", 0.0) * 0.5)
        confidence = float(max(prob_dict.values()))

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "shield_decision": decision,
            "risk_score": round(risk_score, 4),
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "violation_reasons": violations,
            "category_scores": {k: round(features[k], 4) for k in self.feature_names},
            "backend": self.backend,
            "latency_ms": round(latency_ms, 3),
        }

    def predict_batch(self, items: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Batch inference pipeline with vectorized operations."""
        return [self.predict(item) for item in items]

    def get_metadata(self) -> Dict[str, Any]:
        """Return engine metadata and performance characteristics."""
        return {
            "backend": self.backend,
            "feature_names": self.feature_names,
            "classes": CLASS_NAMES,
            "model_metadata": self.metadata,
        }


_engine_instance: Optional[EvalShieldEngine] = None
_engine_lock = threading.Lock()


def get_inference_engine(artifacts_dir: Optional[Path | str] = None) -> EvalShieldEngine:
    """Thread-safe singleton getter for the inference engine."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = EvalShieldEngine(artifacts_dir=artifacts_dir)
    return _engine_instance
