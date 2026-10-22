"""Unit tests for Strands Decider 2B client and integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import requests

from devflow.decider import (
    DECIDER_MODEL_ID,
    DeciderResult,
    StrandsDeciderClient,
    VALID_TOOLS,
)


class TestStrandsDeciderClient:
    def setup_method(self):
        self.client = StrandsDeciderClient(endpoint_url="http://127.0.0.1:8099")

    def test_build_systemone_payload(self):
        payload = self.client.build_systemone_payload("Test state content")
        assert payload["state"] == "Test state content"
        assert payload["model"] == DECIDER_MODEL_ID
        assert "tool_choice" in payload["questions"]
        q = payload["questions"]["tool_choice"]
        assert q["type"] == "choice"
        assert set(q["criteria"].keys()) == set(VALID_TOOLS.keys())

    @patch("requests.post")
    def test_successful_decider_inference(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": DECIDER_MODEL_ID,
            "answers": {
                "tool_choice": {
                    "type": "choice",
                    "choice": "debug_error",
                    "probabilities": {
                        "explain_code": 0.05,
                        "debug_error": 0.85,
                        "analyze_test": 0.05,
                        "search_docs": 0.05,
                    },
                    "confidence": 0.80,
                }
            },
            "latency_ms": 42.1,
        }
        mock_post.return_value = mock_resp

        result = self.client.select_tool("Why is this crashing?", "Traceback...")
        assert result.selected_tool == "debug_error"
        assert result.confidence == 0.80
        assert result.is_live is True
        assert result.status == "success"
        assert result.probabilities["debug_error"] == 0.85

    @patch("requests.post")
    def test_invalid_choice_recovery(self, mock_post):
        # Model returns an unexpected choice name not in VALID_TOOLS
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": DECIDER_MODEL_ID,
            "answers": {
                "tool_choice": {
                    "type": "choice",
                    "choice": "hallucinated_tool_xyz",
                    "probabilities": {
                        "explain_code": 0.70,
                        "debug_error": 0.10,
                        "analyze_test": 0.10,
                        "search_docs": 0.10,
                    },
                    "confidence": 0.60,
                }
            },
        }
        mock_post.return_value = mock_resp

        result = self.client.select_tool("Check this code", "")
        # Should safely fall back to the highest probability valid tool (explain_code)
        assert result.selected_tool in VALID_TOOLS
        assert result.selected_tool == "explain_code"

    @patch("requests.post")
    def test_server_connection_timeout_fallback(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectTimeout("Connection timed out")

        # Fallback enabled
        result = self.client.select_tool("KeyError in traceback", "Traceback...", allow_offline_fallback=True)
        assert result.is_live is False
        assert "offline_fallback" in result.status
        assert result.selected_tool == "debug_error"
        assert result.confidence > 0.0

    @patch("requests.post")
    def test_server_error_without_fallback(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = self.client.select_tool("KeyError", "", allow_offline_fallback=False)
        assert result.is_live is False
        assert result.status == "connection_error"
        assert result.confidence == 0.0

    @patch("requests.get")
    def test_check_health_online(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "model": DECIDER_MODEL_ID}
        mock_get.return_value = mock_resp

        health = self.client.check_health()
        assert health["online"] is True
        assert health["details"]["status"] == "ok"

    @patch("requests.get")
    def test_check_health_offline(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Offline")
        health = self.client.check_health()
        assert health["online"] is False
        assert health["error"] is not None
