"""Unit tests for DevFlow AI Router and Decision Policy Engine."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from devflow.bedrock_client import BedrockClient, BedrockResponse
from devflow.decider import DeciderResult, StrandsDeciderClient
from devflow.router import DevFlowRouter, DevFlowResult


class TestDevFlowRouter:
    def setup_method(self):
        self.mock_decider = MagicMock(spec=StrandsDeciderClient)
        self.mock_bedrock = MagicMock(spec=BedrockClient)
        self.mock_bedrock.model_id = "test-bedrock-model"
        self.mock_bedrock.region = "us-east-1"
        self.mock_bedrock.generate_explanation.return_value = BedrockResponse(
            explanation="Synthesized explanation based on tool findings.",
            model_id="test-bedrock-model",
            region="us-east-1",
            input_tokens=100,
            output_tokens=50,
            latency_ms=120.0,
            is_simulated=False,
            status="success",
        )
        self.router = DevFlowRouter(
            decider_client=self.mock_decider,
            bedrock_client=self.mock_bedrock,
            confidence_threshold=0.60,
        )

    def test_route_explain_code(self):
        self.mock_decider.select_tool.return_value = DeciderResult(
            selected_tool="explain_code",
            confidence=0.92,
            probabilities={"explain_code": 0.94, "debug_error": 0.02, "analyze_test": 0.02, "search_docs": 0.02},
            latency_ms=35.0,
            model_name="strands-decider-2B",
            is_live=True,
            status="success",
        )

        res: DevFlowResult = self.router.route_and_execute(
            user_request="Explain this quicksort algorithm",
            context="def quicksort(arr): return arr",
        )

        assert res.trace.dispatched_tool == "explain_code"
        assert res.trace.confidence_passed is True
        assert res.tool_output.tool_name == "explain_code"
        assert res.trace.routing_action == "direct_dispatch"
        assert "Synthesized explanation" in res.final_explanation

    def test_route_debug_error(self):
        self.mock_decider.select_tool.return_value = DeciderResult(
            selected_tool="debug_error",
            confidence=0.88,
            probabilities={"explain_code": 0.04, "debug_error": 0.91, "analyze_test": 0.03, "search_docs": 0.02},
            latency_ms=32.0,
            model_name="strands-decider-2B",
            is_live=True,
            status="success",
        )

        res: DevFlowResult = self.router.route_and_execute(
            user_request="Traceback NullPointerException",
            context="Traceback (most recent call last):\nFile a.py, line 2",
        )

        assert res.trace.dispatched_tool == "debug_error"
        assert res.tool_output.tool_name == "debug_error"

    def test_route_analyze_test(self):
        self.mock_decider.select_tool.return_value = DeciderResult(
            selected_tool="analyze_test",
            confidence=0.85,
            probabilities={"explain_code": 0.05, "debug_error": 0.05, "analyze_test": 0.88, "search_docs": 0.02},
            latency_ms=30.0,
            model_name="strands-decider-2B",
            is_live=True,
            status="success",
        )

        res: DevFlowResult = self.router.route_and_execute(
            user_request="Why is the pytest suite red?",
            context="FAILED test_cart.py - AssertionError: assert 1 == 2",
        )

        assert res.trace.dispatched_tool == "analyze_test"
        assert res.tool_output.tool_name == "analyze_test"

    def test_route_search_docs(self):
        self.mock_decider.select_tool.return_value = DeciderResult(
            selected_tool="search_docs",
            confidence=0.81,
            probabilities={"explain_code": 0.05, "debug_error": 0.05, "analyze_test": 0.05, "search_docs": 0.85},
            latency_ms=29.0,
            model_name="strands-decider-2B",
            is_live=True,
            status="success",
        )

        res: DevFlowResult = self.router.route_and_execute(
            user_request="Show me the Bedrock SDK setup guide",
            context="",
        )

        assert res.trace.dispatched_tool == "search_docs"
        assert res.tool_output.tool_name == "search_docs"

    def test_low_confidence_policy_enforcement(self):
        # Confidence is 0.45, below threshold 0.60
        self.mock_decider.select_tool.return_value = DeciderResult(
            selected_tool="explain_code",
            confidence=0.45,
            probabilities={"explain_code": 0.40, "debug_error": 0.25, "analyze_test": 0.20, "search_docs": 0.15},
            latency_ms=38.0,
            model_name="strands-decider-2B",
            is_live=True,
            status="success",
        )

        res: DevFlowResult = self.router.route_and_execute(
            user_request="Vague or uncertain developer question",
            context="",
        )

        assert res.trace.confidence_passed is False
        assert res.trace.routing_action == "low_confidence_clarification"
        assert any("below threshold" in note for note in res.trace.policy_notes)

    def test_empty_input_handling(self):
        res: DevFlowResult = self.router.route_and_execute(user_request="", context="")
        assert res.trace.routing_action == "rejected_empty"
        assert res.trace.confidence_passed is False
        assert "Please provide" in res.final_explanation

    def test_offline_fallback_flagged_in_trace_notes(self):
        self.mock_decider.select_tool.return_value = DeciderResult(
            selected_tool="debug_error",
            confidence=0.88,
            probabilities={"explain_code": 0.04, "debug_error": 0.91, "analyze_test": 0.03, "search_docs": 0.02},
            latency_ms=28.5,
            model_name="strands-decider-2B (simulated_offline)",
            is_live=False,
            status="offline_fallback (Decider server offline)",
        )

        res: DevFlowResult = self.router.route_and_execute(
            user_request="Why is this failing?",
            context="KeyError: 'foo'",
        )

        assert res.trace.decider_is_live is False
        assert any("UNVERIFIED LIVE INFERENCE" in note for note in res.trace.policy_notes)

    def test_strict_live_mode_propagated(self):
        self.mock_decider.select_tool.return_value = DeciderResult(
            selected_tool="explain_code",
            confidence=0.0,
            probabilities={"explain_code": 0.25, "debug_error": 0.25, "analyze_test": 0.25, "search_docs": 0.25},
            latency_ms=10.0,
            model_name="strands-decider-2B",
            is_live=False,
            status="connection_error",
        )

        res: DevFlowResult = self.router.route_and_execute(
            user_request="Explain code",
            context="",
            allow_offline_fallback=False,
        )

        self.mock_decider.select_tool.assert_called_with(
            user_request="Explain code",
            context="",
            allow_offline_fallback=False,
        )
        assert res.trace.decider_is_live is False
        assert res.trace.confidence_passed is False
        assert res.trace.routing_action == "halted_strict_live_error"
        assert res.tool_output.tool_name == "none"
        assert "Strict Live Mode Halted" in res.final_explanation
