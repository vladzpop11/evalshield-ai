"""EvalShield-AI: Champion / Challenger Shadow Deployment Engine.

Provides zero-latency shadow routing for continuous candidate model evaluation.
The production Champion engine processes and returns the authoritative user response.
Concurrently, a configurable percentage of traffic is mirrored asynchronously to the
Challenger engine to detect decision mismatches, probability drift, and latency deltas
without impacting client response time.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .inference import EvalShieldEngine, get_inference_engine

logger = logging.getLogger("evalshield_shadow_router")


class ChampionChallengerRouter:
    """Production Champion/Challenger Router with async shadow evaluation."""

    def __init__(
        self,
        champion_engine: Optional[EvalShieldEngine] = None,
        challenger_engine: Optional[EvalShieldEngine] = None,
        shadow_traffic_rate: Optional[float] = None,
        risk_delta_threshold: float = 0.15,
        discrepancy_log_path: Optional[Path | str] = None,
        max_workers: int = 2,
    ) -> None:
        self.artifacts_dir = Path(__file__).resolve().parent / "artifacts"

        # Champion defaults to primary production engine (INT8 ONNX)
        self.champion = champion_engine or get_inference_engine()

        # Challenger defaults to FP32 ONNX or Joblib as comparison engine
        if challenger_engine is not None:
            self.challenger = challenger_engine
        else:
            try:
                # Attempt to load FP32 ONNX as challenger if champion is INT8
                pref = "onnx" if self.champion.backend == "onnx_int8" else "sklearn_joblib"
                self.challenger = EvalShieldEngine(
                    artifacts_dir=self.artifacts_dir,
                    preferred_backend=pref,
                )
            except Exception as e:
                logger.warning("Could not initialize secondary engine for challenger: %s", e)
                self.challenger = None

        # Shadow configuration
        if shadow_traffic_rate is not None:
            self.shadow_traffic_rate = max(0.0, min(1.0, float(shadow_traffic_rate)))
        else:
            self.shadow_traffic_rate = float(os.getenv("SHADOW_TRAFFIC_RATE", "1.0"))

        self.risk_delta_threshold = risk_delta_threshold

        if discrepancy_log_path is None:
            self.discrepancy_log_path = self.artifacts_dir / "discrepancy_log.jsonl"
        else:
            self.discrepancy_log_path = Path(discrepancy_log_path)

        # Thread pool for asynchronous shadow execution
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="shadow_evaluator",
        )
        self._lock = threading.Lock()

        # Telemetry counters
        self._total_requests: int = 0
        self._shadow_evaluations: int = 0
        self._discrepancy_count: int = 0
        self._decision_mismatch_count: int = 0
        self._risk_delta_breach_count: int = 0
        self._champ_latency_sum_ms: float = 0.0
        self._chal_latency_sum_ms: float = 0.0

        # Fixed-size in-memory ring buffer for recent discrepancies
        self._recent_discrepancies: Deque[Dict[str, Any]] = deque(maxlen=50)

    @property
    def is_challenger_active(self) -> bool:
        """Return True if a secondary challenger engine is loaded."""
        return self.challenger is not None

    def route_and_evaluate(
        self,
        features: Dict[str, float],
        prompt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute synchronous champion prediction and trigger async shadow evaluation.

        Returns the authoritative Champion decision immediately without added latency.
        """
        champ_res = self.champion.predict(features)

        with self._lock:
            self._total_requests += 1
            self._champ_latency_sum_ms += champ_res.get("latency_ms", 0.0)

        # Determine if this request qualifies for shadow evaluation
        if self.challenger is not None and self.shadow_traffic_rate > 0.0:
            if random.random() < self.shadow_traffic_rate:
                self._executor.submit(
                    self._execute_shadow_evaluation,
                    features=dict(features),
                    prompt_id=prompt_id,
                    champ_result=dict(champ_res),
                )

        return champ_res

    def _execute_shadow_evaluation(
        self,
        features: Dict[str, float],
        prompt_id: Optional[str],
        champ_result: Dict[str, Any],
    ) -> None:
        """Run challenger model in the background and record divergence metrics."""
        if self.challenger is None:
            return

        try:
            chal_result = self.challenger.predict(features)
        except Exception as exc:
            logger.error("Challenger shadow prediction failed: %s", exc)
            return

        champ_decision = champ_result.get("shield_decision")
        chal_decision = chal_result.get("shield_decision")
        decision_mismatch = bool(champ_decision != chal_decision)

        champ_risk = champ_result.get("risk_score", 0.0)
        chal_risk = chal_result.get("risk_score", 0.0)
        risk_delta = round(abs(champ_risk - chal_risk), 4)
        risk_breach = bool(risk_delta > self.risk_delta_threshold)

        is_discrepancy = decision_mismatch or risk_breach

        with self._lock:
            self._shadow_evaluations += 1
            self._chal_latency_sum_ms += chal_result.get("latency_ms", 0.0)

            if is_discrepancy:
                self._discrepancy_count += 1
                if decision_mismatch:
                    self._decision_mismatch_count += 1
                if risk_breach:
                    self._risk_delta_breach_count += 1

                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_id": prompt_id,
                    "champion_backend": champ_result.get("backend"),
                    "challenger_backend": chal_result.get("backend"),
                    "champion_decision": champ_decision,
                    "challenger_decision": chal_decision,
                    "decision_mismatch": decision_mismatch,
                    "champion_risk_score": champ_risk,
                    "challenger_risk_score": chal_risk,
                    "risk_delta": risk_delta,
                    "champion_latency_ms": champ_result.get("latency_ms"),
                    "challenger_latency_ms": chal_result.get("latency_ms"),
                    "category_scores": features,
                }
                self._recent_discrepancies.append(record)
                self._write_discrepancy_log(record)

    def _write_discrepancy_log(self, record: Dict[str, Any]) -> None:
        """Append discrepancy record to JSONL log file thread-safely."""
        try:
            self.discrepancy_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.discrepancy_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("Failed to append discrepancy log: %s", exc)

    def get_shadow_stats(self) -> Dict[str, Any]:
        """Return real-time shadow evaluation and telemetry statistics."""
        with self._lock:
            shadow_count = self._shadow_evaluations
            agreement_rate = (
                round(1.0 - (self._decision_mismatch_count / shadow_count), 4)
                if shadow_count > 0
                else 1.0
            )
            avg_champ_latency = (
                round(self._champ_latency_sum_ms / max(1, self._total_requests), 3)
            )
            avg_chal_latency = (
                round(self._chal_latency_sum_ms / max(1, shadow_count), 3)
                if shadow_count > 0
                else 0.0
            )

            return {
                "champion_backend": self.champion.backend,
                "challenger_backend": self.challenger.backend if self.challenger else None,
                "challenger_active": self.is_challenger_active,
                "shadow_traffic_rate": self.shadow_traffic_rate,
                "risk_delta_threshold": self.risk_delta_threshold,
                "total_production_requests": self._total_requests,
                "shadow_evaluations_run": shadow_count,
                "discrepancy_count": self._discrepancy_count,
                "decision_mismatches": self._decision_mismatch_count,
                "risk_delta_breaches": self._risk_delta_breach_count,
                "decision_agreement_rate": agreement_rate,
                "avg_champion_latency_ms": avg_champ_latency,
                "avg_challenger_latency_ms": avg_chal_latency,
            }

    def get_recent_discrepancies(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent discrepancies recorded."""
        with self._lock:
            items = list(self._recent_discrepancies)
            return items[-limit:]

    def reset_stats(self) -> None:
        """Reset internal metrics counters."""
        with self._lock:
            self._total_requests = 0
            self._shadow_evaluations = 0
            self._discrepancy_count = 0
            self._decision_mismatch_count = 0
            self._risk_delta_breach_count = 0
            self._champ_latency_sum_ms = 0.0
            self._chal_latency_sum_ms = 0.0
            self._recent_discrepancies.clear()

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully terminate background thread pool."""
        self._executor.shutdown(wait=wait)


_router_instance: Optional[ChampionChallengerRouter] = None
_router_lock = threading.Lock()


def get_champion_challenger_router() -> ChampionChallengerRouter:
    """Thread-safe singleton getter for Champion/Challenger router."""
    global _router_instance
    if _router_instance is None:
        with _router_lock:
            if _router_instance is None:
                _router_instance = ChampionChallengerRouter()
    return _router_instance
