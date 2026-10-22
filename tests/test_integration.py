"""Integration tests for DevFlow AI.

Verifies end-to-end communication across Strands Decider 2B System One API,
tool execution, and Amazon Bedrock response synthesis.
"""

from __future__ import annotations

import os
import threading
import time
import pytest
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from devflow.bedrock_client import BedrockClient
from devflow.decider import (
    DECIDER_MODEL_ID,
    StrandsDeciderClient,
    VALID_TOOLS,
)
from devflow.router import DevFlowRouter, DevFlowResult


# --- Mock Strands Decider System One Test Server ---

def create_mock_decider_app() -> FastAPI:
    """Creates a lightweight FastAPI app implementing the official System One API."""
    app = FastAPI(title="Mock Strands Decider 2B")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": DECIDER_MODEL_ID,
            "checkpoint": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
            "device": "mock-cpu",
        }

    @app.post("/v1/systemone")
    def systemone(req: dict):
        state = req.get("state", "").lower()

        # Deterministic routing based on state keywords
        if any(w in state for w in ["traceback", "keyerror", "nullpointer", "error"]):
            choice = "debug_error"
            probs = {"explain_code": 0.05, "debug_error": 0.85, "analyze_test": 0.05, "search_docs": 0.05}
            conf = 0.80
        elif any(w in state for w in ["assert", "pytest", "failed", "unittest"]):
            choice = "analyze_test"
            probs = {"explain_code": 0.05, "debug_error": 0.05, "analyze_test": 0.85, "search_docs": 0.05}
            conf = 0.80
        elif any(w in state for w in ["documentation", "docs", "guide", "specs"]):
            choice = "search_docs"
            probs = {"explain_code": 0.05, "debug_error": 0.05, "analyze_test": 0.05, "search_docs": 0.85}
            conf = 0.80
        else:
            choice = "explain_code"
            probs = {"explain_code": 0.85, "debug_error": 0.05, "analyze_test": 0.05, "search_docs": 0.05}
            conf = 0.80

        return JSONResponse(
            content={
                "model": DECIDER_MODEL_ID,
                "answers": {
                    "tool_choice": {
                        "type": "choice",
                        "choice": choice,
                        "probabilities": probs,
                        "confidence": conf,
                    }
                },
                "usage": {"input_tokens": 64, "output_tokens": 0},
                "latency_ms": 31.4,
            }
        )

    return app


@pytest.fixture(scope="module")
def mock_decider_server():
    """Starts a real HTTP System One server on a local test port."""
    app = create_mock_decider_app()
    port = 8199
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to bind
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True


class TestEndToEndIntegration:
    @pytest.mark.integration
    def test_full_pipeline_via_systemone_http(self, mock_decider_server):
        """Tests end-to-end request flow over real HTTP System One API."""
        client = StrandsDeciderClient(endpoint_url=mock_decider_server)
        router = DevFlowRouter(
            decider_client=client,
            bedrock_client=BedrockClient(),
            confidence_threshold=0.60,
        )

        result: DevFlowResult = router.route_and_execute(
            user_request="Why is my dictionary throwing a KeyError?",
            context='Traceback (most recent call last):\n  File "run.py", line 12\nKeyError: \'id\'',
        )

        trace = result.trace
        assert trace.decider_is_live is True
        assert trace.decider_status == "success"
        assert trace.dispatched_tool == "debug_error"
        assert trace.confidence >= 0.60
        assert trace.confidence_passed is True
        assert result.tool_output.success is True
        assert "KeyError" in result.tool_output.metadata["exceptions_found"]
        assert len(result.final_explanation) > 50

    @pytest.mark.integration
    def test_live_decider_server_if_available(self):
        """Verifies live inference if an official strands-decider server is actively running.

        If port 8099 is offline, records the exact hardware/model prerequisite blocker.
        """
        live_url = os.environ.get("STRANDS_DECIDER_URL", "http://localhost:8099")
        client = StrandsDeciderClient(endpoint_url=live_url)
        health = client.check_health()

        if not health["online"]:
            pytest.skip(
                f"Live Decider 2B inference server offline at {live_url}. "
                "Hardware prerequisite: 5GB VRAM (CUDA/Metal) and Strands Decider 2B checkpoint. "
                "To start: `strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099`"
            )

        # If live server is online, run actual inference
        res = client.select_tool("Explain this Python class", "class Agent: pass")
        assert res.is_live is True
        assert res.selected_tool in VALID_TOOLS
        assert res.confidence >= 0.0
