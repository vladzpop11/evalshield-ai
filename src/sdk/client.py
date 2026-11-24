"""EvalShield-AI: Python Client SDK.

Lightweight, high-performance client for interacting with the EvalShield-AI API.
Supports synchronous and asynchronous evaluation, PII redaction, and healthchecks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from src.ml.pii_detector import get_pii_detector

logger = logging.getLogger("evalshield_sdk")


class EvalShieldException(Exception):
    """Base exception for EvalShield client errors."""
    pass


class EvalShieldBlockedException(EvalShieldException):
    """Raised when an AI interaction is blocked by safety guardrails."""

    def __init__(
        self,
        decision: str,
        risk_score: float,
        violations: List[str],
        prompt_id: Optional[str] = None,
    ) -> None:
        self.decision = decision
        self.risk_score = risk_score
        detector = get_pii_detector()
        self.violations = [detector.redact_pii(str(item))[0] for item in violations]
        self.prompt_id = detector.redact_pii(prompt_id)[0] if prompt_id else None
        super().__init__(
            f"EvalShield Guardrail BLOCKED interaction (risk={risk_score:.2f}). "
            f"Violations: {'; '.join(self.violations)}"
        )


class EvalShieldClient:
    """Official Python Client for EvalShield-AI Guardrail Platform."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 5.0,
        app: Optional[Any] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.app = app  # Allows passing FastAPI app directly for testing via ASGITransport
        self._api_key = api_key

    def _auth_headers(self) -> Dict[str, str]:
        """Return authentication headers without logging or formatting the key elsewhere."""
        if self._api_key is None:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    def _get_async_transport(self) -> Optional[httpx.BaseTransport]:
        if self.app is not None:
            return httpx.ASGITransport(app=self.app)
        return None

    def evaluate(
        self,
        prompt_text: Optional[str] = None,
        response_text: Optional[str] = None,
        prompt_id: Optional[str] = None,
        redact_pii: bool = False,
        canary_tokens: Optional[List[str]] = None,
        toxicity_score: Optional[float] = None,
        hallucination_risk: Optional[float] = None,
        prompt_injection_score: Optional[float] = None,
        semantic_drift: Optional[float] = None,
        token_entropy: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute synchronous guardrail evaluation."""
        payload: Dict[str, Any] = {
            "prompt_id": prompt_id,
            "redact_pii": redact_pii,
        }
        if prompt_text is not None:
            payload["prompt_text"] = prompt_text
        if response_text is not None:
            payload["response_text"] = response_text
        if canary_tokens is not None:
            payload["canary_tokens"] = canary_tokens
        if toxicity_score is not None:
            payload["toxicity_score"] = toxicity_score
        if hallucination_risk is not None:
            payload["hallucination_risk"] = hallucination_risk
        if prompt_injection_score is not None:
            payload["prompt_injection_score"] = prompt_injection_score
        if semantic_drift is not None:
            payload["semantic_drift"] = semantic_drift
        if token_entropy is not None:
            payload["token_entropy"] = token_entropy

        if self.app is not None:
            from starlette.testclient import TestClient
            with TestClient(self.app, base_url=self.base_url) as client:
                resp = client.post("/evaluate", json=payload, headers=self._auth_headers())
                if resp.status_code != 200:
                    raise EvalShieldException(
                        f"EvalShield API request failed ({resp.status_code})."
                    )
                return resp.json()

        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post("/evaluate", json=payload, headers=self._auth_headers())
            if resp.status_code != 200:
                raise EvalShieldException(
                    f"EvalShield API request failed ({resp.status_code})."
                )
            return resp.json()

    async def async_evaluate(
        self,
        prompt_text: Optional[str] = None,
        response_text: Optional[str] = None,
        prompt_id: Optional[str] = None,
        redact_pii: bool = False,
        canary_tokens: Optional[List[str]] = None,
        toxicity_score: Optional[float] = None,
        hallucination_risk: Optional[float] = None,
        prompt_injection_score: Optional[float] = None,
        semantic_drift: Optional[float] = None,
        token_entropy: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute asynchronous guardrail evaluation."""
        payload: Dict[str, Any] = {
            "prompt_id": prompt_id,
            "redact_pii": redact_pii,
        }
        if prompt_text is not None:
            payload["prompt_text"] = prompt_text
        if response_text is not None:
            payload["response_text"] = response_text
        if canary_tokens is not None:
            payload["canary_tokens"] = canary_tokens
        if toxicity_score is not None:
            payload["toxicity_score"] = toxicity_score
        if hallucination_risk is not None:
            payload["hallucination_risk"] = hallucination_risk
        if prompt_injection_score is not None:
            payload["prompt_injection_score"] = prompt_injection_score
        if semantic_drift is not None:
            payload["semantic_drift"] = semantic_drift
        if token_entropy is not None:
            payload["token_entropy"] = token_entropy

        transport = self._get_async_transport()
        async with httpx.AsyncClient(base_url=self.base_url, transport=transport, timeout=self.timeout) as client:
            resp = await client.post("/evaluate", json=payload, headers=self._auth_headers())
            if resp.status_code != 200:
                raise EvalShieldException(
                    f"EvalShield API request failed ({resp.status_code})."
                )
            return resp.json()

    def check_health(self) -> bool:
        """Check if EvalShield service is operational."""
        try:
            if self.app is not None:
                from starlette.testclient import TestClient
                with TestClient(self.app, base_url=self.base_url) as client:
                    res = client.get("/health", headers=self._auth_headers())
                    return res.status_code == 200 and res.json().get("status") == "healthy"

            with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
                res = client.get("/health", headers=self._auth_headers())
                return res.status_code == 200 and res.json().get("status") == "healthy"
        except Exception:
            return False

    def simulate_red_team(
        self,
        seed_prompt: str,
        rounds: int = 10,
        mutation_types: Optional[List[str]] = None,
        trigger_recalibration: bool = True,
    ) -> Dict[str, Any]:
        """Execute an automated adversarial red-team campaign synchronously."""
        payload = {
            "seed_prompt": seed_prompt,
            "rounds": rounds,
            "mutation_types": mutation_types,
            "trigger_recalibration": trigger_recalibration,
        }
        if self.app is not None:
            from starlette.testclient import TestClient
            with TestClient(self.app, base_url=self.base_url) as client:
                res = client.post("/red-team/simulate", json=payload, headers=self._auth_headers())
                if res.status_code != 200:
                    raise EvalShieldException(f"Red-team simulation failed ({res.status_code}).")
                return res.json()

        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            res = client.post("/red-team/simulate", json=payload, headers=self._auth_headers())
            if res.status_code != 200:
                raise EvalShieldException(f"Red-team simulation failed ({res.status_code}).")
            return res.json()

    async def async_simulate_red_team(
        self,
        seed_prompt: str,
        rounds: int = 10,
        mutation_types: Optional[List[str]] = None,
        trigger_recalibration: bool = True,
    ) -> Dict[str, Any]:
        """Execute an automated adversarial red-team campaign asynchronously."""
        payload = {
            "seed_prompt": seed_prompt,
            "rounds": rounds,
            "mutation_types": mutation_types,
            "trigger_recalibration": trigger_recalibration,
        }
        transport = self._get_async_transport()
        async with httpx.AsyncClient(base_url=self.base_url, transport=transport, timeout=self.timeout) as client:
            res = await client.post("/red-team/simulate", json=payload, headers=self._auth_headers())
            if res.status_code != 200:
                raise EvalShieldException(f"Red-team simulation failed ({res.status_code}).")
            return res.json()
