"""Strands Decider 2B Client & Integration.

Integrates with the official Strands Decider System One API:
- Model: StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
- Protocol: HTTP /v1/systemone and /health (default port 8099)
- Schema: Typed ChoiceQuestion with calibrated confidence and probability distribution.

Reference: https://github.com/strands-labs/strands-decider
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger("devflow.decider")

DEFAULT_DECIDER_URL = os.environ.get("STRANDS_DECIDER_URL", "http://localhost:8099")
DECIDER_MODEL_ID = os.environ.get(
    "STRANDS_DECIDER_MODEL_ID",
    "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
)

# The four valid developer tools
VALID_TOOLS = {
    "explain_code": "Explain code structure, syntax, logic, algorithms, or programming concepts.",
    "debug_error": "Analyze exceptions, stack traces, crash logs, and runtime error messages.",
    "analyze_test": "Interpret failing test output, assertions, test runner logs, and expected vs actual values.",
    "search_docs": "Retrieve relevant documentation, API specifications, and architectural guides from the local documentation collection.",
}

TOOL_SELECTION_INSTRUCTIONS = (
    "Select the most appropriate developer tool to process the user's request."
)


@dataclass
class DeciderResult:
    """Standardized decision outcome from Strands Decider 2B."""
    selected_tool: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_name: str
    is_live: bool
    status: str
    raw_response: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_tool": self.selected_tool,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "latency_ms": round(self.latency_ms, 2),
            "model_name": self.model_name,
            "is_live": self.is_live,
            "status": self.status,
            "error_message": self.error_message,
        }


DEFAULT_DECIDER_API_KEY = os.environ.get("STRANDS_DECIDER_API_KEY")


class StrandsDeciderClient:
    """Client for Strands Decider 2B System One inference."""

    def __init__(
        self,
        endpoint_url: str = DEFAULT_DECIDER_URL,
        timeout_seconds: float = 5.0,
        api_key: str | None = None,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_key = (api_key if api_key is not None else DEFAULT_DECIDER_API_KEY) or None

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"
        return headers

    def _get_timeouts(self) -> tuple[float, float]:
        """Adaptive timeouts: fast fail for localhost, reasonable handshake time for remote tunnels."""
        is_local = "localhost" in self.endpoint_url or "127.0.0.1" in self.endpoint_url
        connect_timeout = 0.5 if is_local else 2.5
        return (connect_timeout, self.timeout_seconds)

    def check_health(self) -> dict[str, Any]:
        """Check whether the local or remote Strands Decider inference server is online."""
        try:
            connect_to, _ = self._get_timeouts()
            resp = requests.get(
                f"{self.endpoint_url}/health",
                headers=self._get_headers(),
                timeout=(connect_to, 2.0),
            )
            if resp.status_code == 200:
                data = resp.json()
                return {"online": True, "details": data, "error": None}
            elif resp.status_code in {401, 403}:
                return {
                    "online": False,
                    "details": None,
                    "error": f"HTTP {resp.status_code} Unauthorized: Check STRANDS_DECIDER_API_KEY.",
                }
            return {
                "online": False,
                "details": None,
                "error": f"HTTP {resp.status_code}: {resp.text[:100]}",
            }
        except Exception as exc:
            return {"online": False, "details": None, "error": str(exc)}

    def build_systemone_payload(self, state: str) -> dict[str, Any]:
        """Build the official typed ChoiceQuestion wire payload for POST /v1/systemone."""
        return {
            "state": state,
            "model": DECIDER_MODEL_ID,
            "questions": {
                "tool_choice": {
                    "type": "choice",
                    "instructions": TOOL_SELECTION_INSTRUCTIONS,
                    "criteria": VALID_TOOLS,
                }
            },
        }

    def select_tool(
        self,
        user_request: str,
        context: str = "",
        allow_offline_fallback: bool = True,
    ) -> DeciderResult:
        """Route a developer request to the appropriate tool using Strands Decider 2B.

        Attempts live inference against the Strands Decider HTTP endpoint (/v1/systemone).
        If unavailable and offline fallback is enabled, evaluates using an offline
        calibrated heuristic model clearly marked as simulated.
        """
        combined_state = user_request.strip()
        if context.strip():
            combined_state = f"{combined_state}\n\nContext:\n{context.strip()}"

        payload = self.build_systemone_payload(combined_state)
        endpoint = f"{self.endpoint_url}/v1/systemone"

        start_time = time.perf_counter()
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=self._get_headers(),
                timeout=self._get_timeouts(),
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if response.status_code == 200:
                body = response.json()
                return self._parse_systemone_response(body, elapsed_ms)
            else:
                err_msg = f"Decider HTTP {response.status_code}: {response.text[:150]}"
                logger.warning(err_msg)
                if allow_offline_fallback:
                    return self._offline_simulated_decision(
                        combined_state, fallback_reason=err_msg
                    )
                return DeciderResult(
                    selected_tool="explain_code",
                    confidence=0.0,
                    probabilities={tool: 0.25 for tool in VALID_TOOLS},
                    latency_ms=elapsed_ms,
                    model_name=DECIDER_MODEL_ID,
                    is_live=False,
                    status="http_error",
                    error_message=err_msg,
                )

        except requests.exceptions.RequestException as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Connection failed to Decider server at {self.endpoint_url}: {exc}"
            logger.info(f"{err_msg}. Triggering offline decision engine.")

            if allow_offline_fallback:
                return self._offline_simulated_decision(
                    combined_state, fallback_reason="Decider server offline"
                )

            return DeciderResult(
                selected_tool="explain_code",
                confidence=0.0,
                probabilities={tool: 0.25 for tool in VALID_TOOLS},
                latency_ms=elapsed_ms,
                model_name=DECIDER_MODEL_ID,
                is_live=False,
                status="connection_error",
                error_message=err_msg,
            )

    def _parse_systemone_response(
        self, body: dict[str, Any], elapsed_ms: float
    ) -> DeciderResult:
        """Parse and validate the official SystemOneResponse structure."""
        answers = body.get("answers", {})
        tool_ans = answers.get("tool_choice")

        if not tool_ans or tool_ans.get("type") != "choice":
            raise ValueError(f"Invalid SystemOne answer payload: {answers}")

        selected = tool_ans.get("choice", "").strip()
        probabilities = tool_ans.get("probabilities", {})
        confidence = float(tool_ans.get("confidence", 0.0))
        model_name = body.get("model", DECIDER_MODEL_ID)

        # Validate that choice is within allowed options
        if selected not in VALID_TOOLS:
            # If invalid tool returned, pick highest probability valid tool
            valid_probs = {k: v for k, v in probabilities.items() if k in VALID_TOOLS}
            if valid_probs:
                selected = max(valid_probs.items(), key=lambda kv: kv[1])[0]
            else:
                selected = "explain_code"

        return DeciderResult(
            selected_tool=selected,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=body.get("latency_ms", elapsed_ms),
            model_name=model_name,
            is_live=True,
            status="success",
            raw_response=body,
        )

    def _offline_simulated_decision(
        self, state: str, fallback_reason: str
    ) -> DeciderResult:
        """Offline deterministic decision simulator mimicking Decider 2B behavior.

        Used when the local GPU server is not actively running (e.g. in test/CI
        or low-memory environments). Explicitly flagged as `is_live=False`.
        Calculates calibrated confidence using the official formula:
            (N * p_max - 1) / (N - 1)
        """
        start = time.perf_counter()
        lower = state.lower()

        # Feature scoring over prompt features
        scores = {tool: 1.0 for tool in VALID_TOOLS}

        # Debugging patterns
        error_indicators = [
            "error", "exception", "traceback", "stack trace", "crash", "nullpointer",
            "keyerror", "typeerror", "valueerror", "indexerror", "failed with code",
            "errno", "syntaxerror", "zero_division", "segmentation fault", "panic"
        ]
        for term in error_indicators:
            if term in lower:
                scores["debug_error"] += 4.5

        # Test patterns
        test_indicators = [
            "pytest", "unittest", "test case", "test failure", "assertionerror",
            "assert ", "expected vs actual", "tests failed", "failing test",
            "test_suite", "jest", "rspec", "spec failure"
        ]
        for term in test_indicators:
            if term in lower:
                scores["analyze_test"] += 5.0

        # Docs search patterns
        docs_indicators = [
            "documentation", "doc", "docs", "guide", "manual", "api reference",
            "reference", "specification", "specs", "configure", "configuration",
            "how do i configure", "how does strands work", "what is bedrock",
            "architectural guide", "lookup", "how to install", "cli options",
            "arguments", "parameters", "search docs"
        ]
        for term in docs_indicators:
            if term in lower:
                scores["search_docs"] += 4.5

        # Code explanation patterns
        code_indicators = [
            "explain", "how does this code work", "what does this function do",
            "breakdown", "walk me through", "def ", "class ", "function",
            "lambda", "async def", "syntax", "refactor explanation"
        ]
        for term in code_indicators:
            if term in lower:
                scores["explain_code"] += 3.5

        # If traceback frames present, strongly boost debug_error
        if 'file "' in lower and "line " in lower:
            scores["debug_error"] += 6.0
        # If pytest output header present
        if "=== failures ===" in lower or "=== fail" in lower:
            scores["analyze_test"] += 7.0

        # Softmax computation
        import math
        exp_scores = {k: math.exp(v) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        probabilities = {k: v / total_exp for k, v in exp_scores.items()}

        selected = max(probabilities.items(), key=lambda kv: kv[1])[0]
        p_max = probabilities[selected]

        # Official Decider 2B confidence formula: (N * p_max - 1) / (N - 1)
        n = len(VALID_TOOLS)
        confidence = max(0.0, min(1.0, (n * p_max - 1.0) / (n - 1.0)))

        # Simulate ~28ms decision latency
        elapsed_ms = (time.perf_counter() - start) * 1000.0 + 28.5

        return DeciderResult(
            selected_tool=selected,
            confidence=confidence,
            probabilities=probabilities,
            latency_ms=elapsed_ms,
            model_name=f"{DECIDER_MODEL_ID} (simulated_offline)",
            is_live=False,
            status=f"offline_fallback ({fallback_reason})",
            raw_response={"simulated": True, "reason": fallback_reason},
        )
