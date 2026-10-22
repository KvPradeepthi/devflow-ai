"""Unit tests for Amazon Bedrock generative client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from devflow.bedrock_client import BedrockClient, BedrockResponse


class TestBedrockClient:
    def setup_method(self):
        self.client = BedrockClient(
            region="us-east-1",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
        )

    @patch("boto3.client")
    def test_bedrock_converse_success(self, mock_boto_client):
        mock_runtime = MagicMock()
        mock_boto_client.return_value = mock_runtime
        mock_runtime.converse.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Here is the concise explanation of the code."}],
                }
            },
            "usage": {"inputTokens": 120, "outputTokens": 45},
        }

        res: BedrockResponse = self.client.generate_explanation(
            user_request="Explain this code",
            tool_name="explain_code",
            tool_title="Code Analysis",
            tool_content="Found function calculate_tax",
        )

        assert res.status == "success"
        assert res.is_simulated is False
        assert "concise explanation" in res.explanation
        assert res.input_tokens == 120
        assert res.output_tokens == 45
        assert res.latency_ms > 0

    @patch("boto3.client")
    def test_bedrock_no_credentials_simulation(self, mock_boto_client):
        mock_runtime = MagicMock()
        mock_boto_client.return_value = mock_runtime
        mock_runtime.converse.side_effect = NoCredentialsError()

        res: BedrockResponse = self.client.generate_explanation(
            user_request="Debug this KeyError",
            tool_name="debug_error",
            tool_title="Error & Traceback Diagnostic",
            tool_content="Verified facts: KeyError at line 42",
            allow_simulation=True,
        )

        assert res.is_simulated is True
        assert "simulated_offline" in res.status
        assert "DevFlow AI Error Diagnostic" in res.explanation
        assert "KeyError at line 42" in res.explanation

    @patch("boto3.client")
    def test_bedrock_client_error_handling(self, mock_boto_client):
        mock_runtime = MagicMock()
        mock_boto_client.return_value = mock_runtime
        mock_runtime.converse.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDeniedException", "Message": "Model not enabled"}},
            operation_name="Converse",
        )

        res: BedrockResponse = self.client.generate_explanation(
            user_request="Search docs",
            tool_name="search_docs",
            tool_title="Documentation Search Results",
            tool_content="Matches found in strands.md",
            allow_simulation=True,
        )

        assert res.is_simulated is True
        assert "AccessDeniedException" in res.status
        assert "DevFlow AI Documentation Summary" in res.explanation
