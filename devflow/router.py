"""Request Router & Decision Policy Engine for DevFlow AI.

Coordinates the end-to-end execution flow:
1. Validates and normalizes developer input.
2. Formulates typed ChoiceQuestion for Strands Decider 2B.
3. Evaluates Decider output against configurable confidence policy.
4. Safely dispatches the selected developer tool.
5. Passes grounded findings to Amazon Bedrock for final synthesis.
6. Emits a verifiable, auditable DecisionTrace.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .bedrock_client import BedrockClient, BedrockResponse
from .decider import DeciderResult, StrandsDeciderClient, VALID_TOOLS
from .tools import TOOL_REGISTRY, ToolOutput

logger = logging.getLogger("devflow.router")

DEFAULT_CONFIDENCE_THRESHOLD = 0.60


@dataclass
class DecisionTrace:
    """Comprehensive, auditable execution trace for a developer request."""
    request_id: str
    timestamp: float
    raw_request: str
    has_context: bool
    decider_model: str
    decider_selected_tool: str
    confidence: float
    probabilities: dict[str, float]
    decider_latency_ms: float
    decider_is_live: bool
    decider_status: str
    confidence_threshold: float
    confidence_passed: bool
    dispatched_tool: str
    routing_action: str
    tool_success: bool
    bedrock_model: str
    bedrock_is_simulated: bool
    bedrock_latency_ms: float
    total_latency_ms: float
    policy_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "raw_request": self.raw_request,
            "has_context": self.has_context,
            "decider": {
                "model": self.decider_model,
                "selected_tool": self.decider_selected_tool,
                "confidence": round(self.confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
                "latency_ms": round(self.decider_latency_ms, 2),
                "is_live": self.decider_is_live,
                "status": self.decider_status,
            },
            "policy": {
                "confidence_threshold": self.confidence_threshold,
                "confidence_passed": self.confidence_passed,
                "dispatched_tool": self.dispatched_tool,
                "routing_action": self.routing_action,
                "notes": self.policy_notes,
            },
            "tool": {
                "dispatched_tool": self.dispatched_tool,
                "success": self.tool_success,
            },
            "bedrock": {
                "model": self.bedrock_model,
                "is_simulated": self.bedrock_is_simulated,
                "latency_ms": round(self.bedrock_latency_ms, 2),
            },
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


@dataclass
class DevFlowResult:
    """Complete response returned to the caller or user interface."""
    trace: DecisionTrace
    tool_output: ToolOutput
    bedrock_response: BedrockResponse
    final_explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace.to_dict(),
            "tool_output": self.tool_output.to_dict(),
            "bedrock": self.bedrock_response.to_dict(),
            "final_explanation": self.final_explanation,
        }


class DevFlowRouter:
    """Central router enforcing Decider 2B selection and Bedrock synthesis."""

    def __init__(
        self,
        decider_client: StrandsDeciderClient | None = None,
        bedrock_client: BedrockClient | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.decider = decider_client or StrandsDeciderClient()
        self.bedrock = bedrock_client or BedrockClient()
        self.confidence_threshold = confidence_threshold

    def route_and_execute(
        self,
        user_request: str,
        context: str = "",
        override_threshold: float | None = None,
        allow_offline_fallback: bool = True,
    ) -> DevFlowResult:
        """Execute full DevFlow pipeline: validate -> Decider 2B -> tool -> Bedrock."""
        start_time = time.perf_counter()
        req_id = str(uuid.uuid4())[:8]
        threshold = override_threshold if override_threshold is not None else self.confidence_threshold
        policy_notes: list[str] = []

        # Step 1: Input Validation & Normalization
        clean_request = user_request.strip()
        clean_context = context.strip()
        if not clean_request and not clean_context:
            empty_tool_out = ToolOutput(
                tool_name="none",
                success=False,
                title="Input Required",
                content="Please provide a request or context snippet.",
            )
            empty_trace = DecisionTrace(
                request_id=req_id,
                timestamp=time.time(),
                raw_request="",
                has_context=False,
                decider_model="none",
                decider_selected_tool="none",
                confidence=0.0,
                probabilities={},
                decider_latency_ms=0.0,
                decider_is_live=False,
                decider_status="empty_input",
                confidence_threshold=threshold,
                confidence_passed=False,
                dispatched_tool="none",
                routing_action="rejected_empty",
                tool_success=False,
                bedrock_model=self.bedrock.model_id,
                bedrock_is_simulated=True,
                bedrock_latency_ms=0.0,
                total_latency_ms=0.0,
                policy_notes=["Request and context were both empty."],
            )
            empty_bedrock = BedrockResponse(
                explanation="Please provide a developer question, code snippet, error message, or test log to begin.",
                model_id=self.bedrock.model_id,
                region=self.bedrock.region,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                is_simulated=True,
                status="empty_input",
            )
            return DevFlowResult(
                trace=empty_trace,
                tool_output=empty_tool_out,
                bedrock_response=empty_bedrock,
                final_explanation=empty_bedrock.explanation,
            )

        # Step 2: Query Strands Decider 2B
        decider_res: DeciderResult = self.decider.select_tool(
            user_request=clean_request,
            context=clean_context,
            allow_offline_fallback=allow_offline_fallback,
        )

        selected_tool = decider_res.selected_tool
        confidence = decider_res.confidence

        if not decider_res.is_live:
            policy_notes.append(
                "UNVERIFIED LIVE INFERENCE: Model server was offline or unreachable. "
                f"Decision evaluated via {decider_res.status}."
            )
        else:
            policy_notes.append(
                f"VERIFIED LIVE INFERENCE: Official Strands Decider 2B server evaluated state in {decider_res.latency_ms:.1f}ms."
            )

        # Strict Live Mode: refuse to route if server is down and fallback is disabled
        if not allow_offline_fallback and not decider_res.is_live:
            routing_action = "halted_strict_live_error"
            endpoint_display = getattr(self.decider, "endpoint_url", "http://localhost:8099")
            policy_notes.append(
                f"STRICT LIVE MODE HALT: Refusing to route request because live Decider server is unreachable ({decider_res.error_message or decider_res.status}). Offline simulation is disabled."
            )
            halted_tool_output = ToolOutput(
                tool_name="none",
                success=False,
                title="Strict Live Mode: Server Offline",
                content=f"Tool execution halted. Strands Decider 2B server is offline at {endpoint_display}. Offline fallback is disabled in Strict Live Mode.",
                metadata={"error": decider_res.error_message or decider_res.status},
            )
            halted_trace = DecisionTrace(
                request_id=req_id,
                timestamp=time.time(),
                raw_request=clean_request,
                has_context=bool(clean_context),
                decider_model=decider_res.model_name,
                decider_selected_tool="none",
                confidence=0.0,
                probabilities=decider_res.probabilities,
                decider_latency_ms=decider_res.latency_ms,
                decider_is_live=False,
                decider_status=decider_res.status,
                confidence_threshold=threshold,
                confidence_passed=False,
                dispatched_tool="none",
                routing_action=routing_action,
                tool_success=False,
                bedrock_model=self.bedrock.model_id,
                bedrock_is_simulated=True,
                bedrock_latency_ms=0.0,
                total_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                policy_notes=policy_notes,
            )
            halted_bedrock = BedrockResponse(
                explanation=(
                    "### Strict Live Mode Halted\n\n"
                    "The application is operating in **Strict Live Mode** and refused to make a simulated decision. "
                    f"The Strands Decider 2B server was unreachable at `{endpoint_display}`.\n\n"
                    "To execute genuine live inference, please start the official server on port 8099:\n"
                    "```powershell\n"
                    "strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099\n"
                    "```"
                ),
                model_id=self.bedrock.model_id,
                region=self.bedrock.region,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                is_simulated=True,
                status="strict_live_halted",
            )
            return DevFlowResult(
                trace=halted_trace,
                tool_output=halted_tool_output,
                bedrock_response=halted_bedrock,
                final_explanation=halted_bedrock.explanation,
            )

        # Step 3: Apply Confidence & Policy Guardrails
        confidence_passed = confidence >= threshold
        dispatched_tool = selected_tool
        routing_action = "direct_dispatch"

        if not confidence_passed:
            routing_action = "low_confidence_clarification"
            policy_notes.append(
                f"Confidence ({confidence:.3f}) was below threshold ({threshold:.2f}). "
                f"Decider favored '{selected_tool}', but the system requests clarification or defaults to general search."
            )
            # Safe fallback: if ambiguous, search docs or explain code as safe default
            dispatched_tool = "search_docs" if "search_docs" in decider_res.probabilities else "explain_code"
        else:
            policy_notes.append(
                f"Confidence ({confidence:.3f}) met threshold ({threshold:.2f}). Dispatched '{dispatched_tool}'."
            )

        # Step 4: Dispatch the Selected Tool
        tool_func = TOOL_REGISTRY.get(dispatched_tool, TOOL_REGISTRY["explain_code"])
        try:
            tool_output: ToolOutput = tool_func(clean_request, clean_context)
        except Exception as exc:
            logger.error(f"Error executing tool {dispatched_tool}: {exc}")
            tool_output = ToolOutput(
                tool_name=dispatched_tool,
                success=False,
                title=f"Error Executing {dispatched_tool}",
                content=f"An exception occurred while executing {dispatched_tool}: {exc}",
                metadata={"error": str(exc)},
            )

        # Step 5: Synthesize Grounded Explanation via Amazon Bedrock
        bedrock_res: BedrockResponse = self.bedrock.generate_explanation(
            user_request=clean_request,
            tool_name=dispatched_tool,
            tool_title=tool_output.title,
            tool_content=tool_output.content,
            raw_context=clean_context,
            allow_simulation=allow_offline_fallback,
        )

        final_explanation = bedrock_res.explanation
        if not final_explanation:
            final_explanation = (
                f"### Bedrock Generation Inactive\n\n"
                f"Status: `{bedrock_res.status}`\n\n"
                f"{bedrock_res.error_message or 'No generative response produced. Ensure AWS credentials and model access are configured.'}"
            )

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 6: Construct Auditable Decision Trace
        trace = DecisionTrace(
            request_id=req_id,
            timestamp=time.time(),
            raw_request=clean_request,
            has_context=bool(clean_context),
            decider_model=decider_res.model_name,
            decider_selected_tool=selected_tool,
            confidence=confidence,
            probabilities=decider_res.probabilities,
            decider_latency_ms=decider_res.latency_ms,
            decider_is_live=decider_res.is_live,
            decider_status=decider_res.status,
            confidence_threshold=threshold,
            confidence_passed=confidence_passed,
            dispatched_tool=dispatched_tool,
            routing_action=routing_action,
            tool_success=tool_output.success,
            bedrock_model=bedrock_res.model_id,
            bedrock_is_simulated=bedrock_res.is_simulated,
            bedrock_latency_ms=bedrock_res.latency_ms,
            total_latency_ms=total_elapsed_ms,
            policy_notes=policy_notes,
        )

        return DevFlowResult(
            trace=trace,
            tool_output=tool_output,
            bedrock_response=bedrock_res,
            final_explanation=final_explanation,
        )
