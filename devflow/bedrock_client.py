"""Amazon Bedrock Generative Inference Client.

Integrates with AWS Bedrock Runtime using the modern Converse API:
- Region: AWS_REGION or AWS_DEFAULT_REGION (defaults to us-east-1)
- Model: BEDROCK_MODEL_ID (defaults to anthropic.claude-3-haiku-20240307-v1:0)
- Graceful error handling for missing credentials, permissions, and throttling.
- Grounds generated answers strictly in the tool outputs.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)

logger = logging.getLogger("devflow.bedrock")

DEFAULT_AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
DEFAULT_BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
)


@dataclass
class BedrockResponse:
    """Outcome from Amazon Bedrock generative inference."""
    explanation: str
    model_id: str
    region: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    is_simulated: bool
    status: str
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation": self.explanation,
            "model_id": self.model_id,
            "region": self.region,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "is_simulated": self.is_simulated,
            "status": self.status,
            "error_message": self.error_message,
        }


class BedrockClient:
    """Client for generating developer explanations via Amazon Bedrock."""

    def __init__(
        self,
        region: str = DEFAULT_AWS_REGION,
        model_id: str = DEFAULT_BEDROCK_MODEL_ID,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        self.region = region
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self._credentials_checked = False
        self._has_credentials = False

    def check_credentials(self) -> tuple[bool, str]:
        """Verify whether AWS credentials are configured in the environment."""
        session = boto3.Session()
        creds = session.get_credentials()
        if creds is None:
            return False, "No AWS credentials found (AWS_ACCESS_KEY_ID or ~/.aws/credentials missing)."
        self._has_credentials = True
        return True, "AWS credentials located."

    def _get_runtime_client(self):
        """Lazy initialization of boto3 Bedrock Runtime client."""
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def generate_explanation(
        self,
        user_request: str,
        tool_name: str,
        tool_title: str,
        tool_content: str,
        raw_context: str = "",
        allow_simulation: bool = True,
    ) -> BedrockResponse:
        """Call Amazon Bedrock to generate a concise, grounded developer explanation."""
        start_time = time.perf_counter()

        system_prompt = (
            "You are DevFlow AI, an expert software engineering assistant. "
            "You receive a user developer request along with structured output produced by "
            f"a specialized developer tool ('{tool_name}').\n\n"
            "Guidelines:\n"
            "1. Ground your explanation directly in the provided tool output.\n"
            "2. Keep the response concise, authoritative, and actionable.\n"
            "3. If code or stack traces are provided, provide exact fixes and highlight relevant lines.\n"
            "4. Distinguish verified facts from hypotheses."
        )

        user_content = (
            f"Developer Request: {user_request}\n\n"
            f"Tool Executed: {tool_title} ({tool_name})\n\n"
            f"Tool Diagnostic Findings:\n{tool_content}\n"
        )
        if raw_context.strip():
            user_content += f"\nOriginal User Context:\n{raw_context.strip()}\n"

        # Attempt AWS Bedrock Converse API
        try:
            client = self._get_runtime_client()

            messages = [
                {
                    "role": "user",
                    "content": [{"text": user_content}],
                }
            ]

            inference_config = {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            }

            response = client.converse(
                modelId=self.model_id,
                messages=messages,
                system=[{"text": system_prompt}],
                inferenceConfig=inference_config,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            explanation_text = "".join(b.get("text", "") for b in content_blocks)

            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)

            return BedrockResponse(
                explanation=explanation_text,
                model_id=self.model_id,
                region=self.region,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=elapsed_ms,
                is_simulated=False,
                status="success",
            )

        except (NoCredentialsError, PartialCredentialsError) as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"AWS credentials not configured: {exc}"
            logger.info(f"{err_msg}. Using simulated generative inference.")

            if allow_simulation:
                return self._simulate_grounded_response(
                    user_request, tool_name, tool_title, tool_content, elapsed_ms, err_msg
                )
            return BedrockResponse(
                explanation="",
                model_id=self.model_id,
                region=self.region,
                input_tokens=0,
                output_tokens=0,
                latency_ms=elapsed_ms,
                is_simulated=False,
                status="no_credentials",
                error_message=err_msg,
            )

        except ClientError as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            error_code = exc.response.get("Error", {}).get("Code", "ClientError")
            error_message = exc.response.get("Error", {}).get("Message", str(exc))
            err_msg = f"Bedrock ClientError ({error_code}): {error_message}"
            logger.warning(err_msg)

            if allow_simulation:
                return self._simulate_grounded_response(
                    user_request, tool_name, tool_title, tool_content, elapsed_ms, err_msg
                )
            return BedrockResponse(
                explanation="",
                model_id=self.model_id,
                region=self.region,
                input_tokens=0,
                output_tokens=0,
                latency_ms=elapsed_ms,
                is_simulated=False,
                status=error_code,
                error_message=err_msg,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Unexpected Bedrock error: {exc}"
            logger.error(err_msg)

            if allow_simulation:
                return self._simulate_grounded_response(
                    user_request, tool_name, tool_title, tool_content, elapsed_ms, err_msg
                )
            return BedrockResponse(
                explanation="",
                model_id=self.model_id,
                region=self.region,
                input_tokens=0,
                output_tokens=0,
                latency_ms=elapsed_ms,
                is_simulated=False,
                status="unexpected_error",
                error_message=err_msg,
            )

    def _simulate_grounded_response(
        self,
        request: str,
        tool_name: str,
        tool_title: str,
        tool_content: str,
        latency_ms: float,
        reason: str,
    ) -> BedrockResponse:
        """Produce a grounded generative response when AWS credentials are not active.

        Strictly incorporates tool diagnostic findings. Clearly marked with is_simulated=True.
        """
        simulated_tokens = len(tool_content.split()) + len(request.split()) + 45

        if tool_name == "explain_code":
            explanation = (
                f"### DevFlow AI Code Explanation (Grounded in Tool Output)\n\n"
                f"Based on the static inspection performed by `{tool_title}`:\n\n"
                f"{tool_content}\n\n"
                f"**Architectural Summary:**\n"
                f"The provided snippet encapsulates clean modular execution. "
                f"Key components are structured for testability and clarity. "
                f"Ensure error boundaries are established around external I/O and state mutations."
            )
        elif tool_name == "debug_error":
            explanation = (
                f"### DevFlow AI Error Diagnostic & Remediation\n\n"
                f"Based on traceback inspection from `{tool_title}`:\n\n"
                f"{tool_content}\n\n"
                f"**Recommended Action Plan:**\n"
                f"1. Review the failing frame highlighted above and inspect variable state immediately prior to throwing.\n"
                f"2. Add input validation or defensive guards to prevent unexpected null/None or missing key operations.\n"
                f"3. Run an isolated unit test reproducing this exact stack trace."
            )
        elif tool_name == "analyze_test":
            explanation = (
                f"### DevFlow AI Test Failure Analysis\n\n"
                f"Diagnostic findings from `{tool_title}`:\n\n"
                f"{tool_content}\n\n"
                f"**Resolution Strategy:**\n"
                f"- Compare the actual output against the asserted contract expectations.\n"
                f"- Ensure test data setup is deterministic and isolated from environmental side effects.\n"
                f"- Update assertions if requirements evolved, or fix implementation logic if behavior diverged."
            )
        else:  # search_docs
            explanation = (
                f"### DevFlow AI Documentation Summary\n\n"
                f"Documentation retrieval results from `{tool_title}`:\n\n"
                f"{tool_content}\n\n"
                f"**Developer Key Takeaways:**\n"
                f"The passages above provide the authoritative specification for your request. "
                f"Consult the referenced local document for full API signatures and configuration options."
            )

        return BedrockResponse(
            explanation=explanation,
            model_id=f"{self.model_id} (simulated_offline)",
            region=self.region,
            input_tokens=simulated_tokens,
            output_tokens=len(explanation.split()),
            latency_ms=latency_ms + 140.0,  # simulate Bedrock generation latency
            is_simulated=True,
            status=f"simulated_offline ({reason})",
        )
