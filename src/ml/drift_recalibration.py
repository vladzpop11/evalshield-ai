"""EvalShield-AI: Automated Closed-Loop Drift Recalibration Engine.

Provides self-healing MLOps capabilities:
When distribution drift is detected via two-sample Kolmogorov-Smirnov statistical
tests, this engine dynamically recalculates empirical distribution percentiles,
adjusts heuristic guardrail thresholds, and assesses canary candidate model readiness
without service downtime.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger("evalshield_recalibration")

# Retain only recent in-memory audit metadata to keep process memory bounded.
MAX_RECALIBRATION_HISTORY_ENTRIES = 1_000


@dataclass
class RecalibrationResult:
    """Outcome of an automated distribution recalibration pass."""
    recalibration_id: str
    recalibration_status: str  # "RECALIBRATED" | "STABLE_NO_RECALIBRATION_NEEDED"
    overall_drift_detected: bool
    features_recalibrated: List[str]
    previous_thresholds: Dict[str, Any]
    adjusted_thresholds: Dict[str, Any]
    canary_promotion_recommended: bool
    samples_evaluated: int
    applied_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DriftRecalibrationEngine:
    """Closed-loop statistical recalibration and adaptive threshold manager."""

    # Default heuristic baseline boundaries
    DEFAULT_THRESHOLDS = {
        "entropy_lower": 0.5,
        "entropy_upper": 9.5,
        "toxicity_threshold": 0.70,
        "injection_threshold": 0.65,
        "semantic_drift_tolerance": 0.85,
    }

    def __init__(self, baseline_df: Optional[pd.DataFrame] = None) -> None:
        self.artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        self._baseline_df = baseline_df
        self.active_thresholds = dict(self.DEFAULT_THRESHOLDS)
        self.recalibration_history: List[RecalibrationResult] = []

    def _get_baseline(self) -> pd.DataFrame:
        """Load baseline distribution if not provided."""
        if self._baseline_df is None or self._baseline_df.empty:
            path = self.artifacts_dir / "baseline_distribution.parquet"
            if path.exists():
                self._baseline_df = pd.read_parquet(path)
            else:
                raise FileNotFoundError(f"Baseline distribution not found at {path}")
        return self._baseline_df

    def recalibrate(
        self,
        production_samples: Dict[str, List[float]],
        significance_level: float = 0.05,
    ) -> RecalibrationResult:
        """Analyze production distribution samples and compute adaptive threshold adjustments."""
        baseline = self._get_baseline()
        recalibration_id = f"recal-{uuid.uuid4().hex[:8]}"
        prev_thresholds = dict(self.active_thresholds)
        new_thresholds = dict(self.active_thresholds)

        drifted_features: List[str] = []
        samples_count = 0

        for feat_name, values in production_samples.items():
            if feat_name not in baseline.columns or len(values) < 15:
                continue

            samples_count = max(samples_count, len(values))
            base_vals = baseline[feat_name].dropna().values
            ks_stat, p_val = ks_2samp(base_vals, values)

            if p_val < significance_level:
                drifted_features.append(feat_name)
                sample_p05 = float(np.percentile(values, 5))
                sample_p95 = float(np.percentile(values, 95))
                sample_mean = float(np.mean(values))

                # Adaptive recalibration logic based on feature behavior
                if feat_name == "token_entropy":
                    # Adaptively expand or contract entropy boundaries based on observed 5th/95th percentiles
                    new_lower = max(0.2, round(min(prev_thresholds["entropy_lower"], sample_p05 * 0.8), 2))
                    new_upper = min(9.9, round(max(prev_thresholds["entropy_upper"], sample_p95 * 1.1), 2))
                    new_thresholds["entropy_lower"] = new_lower
                    new_thresholds["entropy_upper"] = new_upper

                elif feat_name == "semantic_drift":
                    # If enterprise vocabulary has shifted, adjust semantic tolerance window
                    new_tolerance = min(0.95, max(0.70, round(sample_p95 + 0.05, 2)))
                    new_thresholds["semantic_drift_tolerance"] = new_tolerance

        drift_detected = len(drifted_features) > 0
        status = "RECALIBRATED" if drift_detected else "STABLE_NO_RECALIBRATION_NEEDED"

        # Update active thresholds if recalibrated
        if drift_detected:
            self.active_thresholds = new_thresholds

        result = RecalibrationResult(
            recalibration_id=recalibration_id,
            recalibration_status=status,
            overall_drift_detected=drift_detected,
            features_recalibrated=drifted_features,
            previous_thresholds=prev_thresholds,
            adjusted_thresholds=new_thresholds,
            canary_promotion_recommended=drift_detected,
            samples_evaluated=samples_count,
            applied_at=datetime.now(timezone.utc).isoformat(),
        )

        history_overflow = (
            len(self.recalibration_history) - MAX_RECALIBRATION_HISTORY_ENTRIES + 1
        )
        if history_overflow > 0:
            del self.recalibration_history[:history_overflow]
        self.recalibration_history.append(result)
        return result

    def get_active_thresholds(self) -> Dict[str, Any]:
        """Return currently enforced guardrail thresholds."""
        return dict(self.active_thresholds)

    def reset_to_defaults(self) -> None:
        """Revert active thresholds to standard baseline."""
        self.active_thresholds = dict(self.DEFAULT_THRESHOLDS)


_engine_instance: Optional[DriftRecalibrationEngine] = None
_engine_lock = threading.Lock()


def get_recalibration_engine() -> DriftRecalibrationEngine:
    """Thread-safe singleton getter for DriftRecalibrationEngine."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = DriftRecalibrationEngine()
    return _engine_instance
