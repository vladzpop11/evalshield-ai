"""EvalShield-AI: Function Decorator and LLM Integration Guard.

Provides zero-boilerplate guardrails for Python functions, LangChain chains,
and OpenAI client wrappers via the `@guard.protect` decorator.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, List, Literal, Optional

from .client import EvalShieldBlockedException, EvalShieldClient


class EvalShieldGuard:
    """Enterprise AI Guardrail Decorator and Workflow Interceptor."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        client: Optional[EvalShieldClient] = None,
        redact_pii: bool = False,
        canary_tokens: Optional[List[str]] = None,
        app: Optional[Any] = None,
    ) -> None:
        self.client = client or EvalShieldClient(base_url=base_url, app=app)
        self.redact_pii = redact_pii
        self.canary_tokens = canary_tokens

    def protect(
        self,
        on_blocked: Literal["raise", "fallback"] = "raise",
        fallback_value: str = "Blocked by EvalShield AI Safety Policy.",
        inspect_output: bool = True,
    ) -> Callable:
        """Decorator to intercept, evaluate, and sanitize AI model calls."""
        def decorator(func: Callable) -> Callable:
            is_async = inspect.iscoroutinefunction(func)

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                prompt_text, arg_idx, kwarg_key = self._extract_prompt(func, args, kwargs)

                eval_res = await self.client.async_evaluate(
                    prompt_text=prompt_text,
                    redact_pii=self.redact_pii,
                    canary_tokens=self.canary_tokens,
                )

                if eval_res.get("shield_decision") == "BLOCKED":
                    if on_blocked == "raise":
                        raise EvalShieldBlockedException(
                            decision="BLOCKED",
                            risk_score=eval_res.get("risk_score", 1.0),
                            violations=eval_res.get("violation_reasons", []),
                        )
                    return fallback_value

                # Sanitize input prompt if PII redaction produced a clean prompt
                sanitized_args, sanitized_kwargs = self._apply_sanitized_prompt(
                    args, kwargs, arg_idx, kwarg_key, eval_res.get("sanitized_prompt")
                )

                # Execute original LLM call
                llm_output = await func(*sanitized_args, **sanitized_kwargs)

                # Output Guardrail Inspection
                if inspect_output and isinstance(llm_output, str):
                    out_res = await self.client.async_evaluate(
                        prompt_text=prompt_text,
                        response_text=llm_output,
                        canary_tokens=self.canary_tokens,
                    )
                    if out_res.get("shield_decision") == "BLOCKED":
                        if on_blocked == "raise":
                            raise EvalShieldBlockedException(
                                decision="BLOCKED",
                                risk_score=out_res.get("risk_score", 1.0),
                                violations=out_res.get("violation_reasons", []),
                            )
                        return fallback_value

                return llm_output

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                prompt_text, arg_idx, kwarg_key = self._extract_prompt(func, args, kwargs)

                eval_res = self.client.evaluate(
                    prompt_text=prompt_text,
                    redact_pii=self.redact_pii,
                    canary_tokens=self.canary_tokens,
                )

                if eval_res.get("shield_decision") == "BLOCKED":
                    if on_blocked == "raise":
                        raise EvalShieldBlockedException(
                            decision="BLOCKED",
                            risk_score=eval_res.get("risk_score", 1.0),
                            violations=eval_res.get("violation_reasons", []),
                        )
                    return fallback_value

                sanitized_args, sanitized_kwargs = self._apply_sanitized_prompt(
                    args, kwargs, arg_idx, kwarg_key, eval_res.get("sanitized_prompt")
                )

                llm_output = func(*sanitized_args, **sanitized_kwargs)

                if inspect_output and isinstance(llm_output, str):
                    out_res = self.client.evaluate(
                        prompt_text=prompt_text,
                        response_text=llm_output,
                        canary_tokens=self.canary_tokens,
                    )
                    if out_res.get("shield_decision") == "BLOCKED":
                        if on_blocked == "raise":
                            raise EvalShieldBlockedException(
                                decision="BLOCKED",
                                risk_score=out_res.get("risk_score", 1.0),
                                violations=out_res.get("violation_reasons", []),
                            )
                        return fallback_value

                return llm_output

            return async_wrapper if is_async else sync_wrapper

        return decorator

    @staticmethod
    def _extract_prompt(
        func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> tuple[Optional[str], Optional[int], Optional[str]]:
        """Identify the prompt string parameter across positional and keyword arguments."""
        # 1. Check common keyword arguments
        for key in ["prompt", "prompt_text", "input_text", "query", "message"]:
            if key in kwargs and isinstance(kwargs[key], str):
                return kwargs[key], None, key

        # 2. Check first positional string argument
        for idx, arg in enumerate(args):
            if isinstance(arg, str):
                return arg, idx, None

        return None, None, None

    @staticmethod
    def _apply_sanitized_prompt(
        args: tuple,
        kwargs: dict,
        arg_idx: Optional[int],
        kwarg_key: Optional[str],
        sanitized_prompt: Optional[str],
    ) -> tuple[tuple, dict]:
        """Replace raw prompt with redacted text if available."""
        if not sanitized_prompt:
            return args, kwargs

        new_args = list(args)
        new_kwargs = dict(kwargs)

        if kwarg_key is not None:
            new_kwargs[kwarg_key] = sanitized_prompt
        elif arg_idx is not None and arg_idx < len(new_args):
            new_args[arg_idx] = sanitized_prompt

        return tuple(new_args), new_kwargs
