"""Live End-to-End Verification Script for DevFlow AI.

Enforces STRICT LIVE MODE (zero simulation):
1. Verifies live Strands Decider 2B server is reachable via GET /health.
2. Verifies active AWS Bedrock credentials and model access.
3. Executes a genuine end-to-end request:
   User Request -> Live Decider 2B -> Dispatched Tool -> Live Bedrock Converse -> Output.

Usage:
    python verify_live_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from devflow.bedrock_client import BedrockClient
from devflow.decider import StrandsDeciderClient, VALID_TOOLS
from devflow.router import DevFlowRouter, DevFlowResult


# Ensure utf-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_live_verification():
    print("=" * 75)
    print("DevFlow AI: Genuine Live End-to-End Verification (Strict Live Mode)")
    print("=" * 75)

    decider_endpoint = os.environ.get("STRANDS_DECIDER_URL", "http://localhost:8099")
    decider_api_key = os.environ.get("STRANDS_DECIDER_API_KEY")
    decider = StrandsDeciderClient(endpoint_url=decider_endpoint, api_key=decider_api_key)

    # 1. Health check Strands Decider
    print(f"\n[1/3] Checking Strands Decider 2B server at {decider_endpoint}...")
    if decider_api_key:
        print("      (Bearer token authentication enabled)")
    health = decider.check_health()
    if not health["online"]:
        print(f"  [-] OFFLINE: Strands Decider server is not reachable at {decider_endpoint}.")
        print(f"      Error: {health['error']}")
        print("\n  [!] Action Required to Enable Live Decider Inference:")
        print("      Run the official model server on a machine with GPU/Metal:")
        print("      strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099")
        decider_live = False
    else:
        details = health.get("details", {})
        print(f"  [+] ONLINE: Strands Decider 2B server is reachable!")
        print(f"      Checkpoint: {details.get('checkpoint', 'unknown')}")
        print(f"      Device: {details.get('device', 'unknown')}")
        decider_live = True

    # 2. Check Amazon Bedrock
    region = os.environ.get("AWS_REGION", "us-east-1")
    model_id = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
    print(f"\n[2/3] Checking Amazon Bedrock configuration (Region: {region}, Model: {model_id})...")
    bedrock = BedrockClient(region=region, model_id=model_id)
    has_creds, creds_msg = bedrock.check_credentials()
    if not has_creds:
        print(f"  [-] NOT CONFIGURED: {creds_msg}")
        print("\n  [!] Action Required to Enable Live Amazon Bedrock Inference:")
        print("      Configure AWS credentials via: aws configure")
        print("      Ensure model access is enabled in the Amazon Bedrock Console.")
        bedrock_live = False
    else:
        print(f"  [+] CONFIGURED: AWS Credentials located!")
        bedrock_live = True

    # 3. Genuine Execution
    print("\n[3/3] Executing Genuine End-to-End Pipeline in STRICT LIVE MODE...")
    if not decider_live:
        print("  [!] ABORTED: Cannot run strict live verification because Strands Decider server is offline.")
        print("      Strict live mode refuses to substitute an offline heuristic simulation.")
        return False

    router = DevFlowRouter(
        decider_client=decider,
        bedrock_client=bedrock,
        confidence_threshold=0.60,
    )

    sample_request = "Why is my dictionary throwing this KeyError?"
    sample_context = (
        'Traceback (most recent call last):\n'
        '  File "service.py", line 42, in get_user\n'
        '    token = payload["auth"]["token"]\n'
        'KeyError: \'auth\''
    )

    print(f"  Request: {sample_request}")
    print("  Context: Python KeyError Traceback at line 42")

    # allow_offline_fallback=False enforces STRICT LIVE MODE
    result: DevFlowResult = router.route_and_execute(
        user_request=sample_request,
        context=sample_context,
        allow_offline_fallback=False,
    )

    trace = result.trace
    print("\n" + "-" * 75)
    print("LIVE EXECUTION RESULTS:")
    print(f"  Live Decider Verified: {trace.decider_is_live}")
    print(f"  Decider Model: {trace.decider_model}")
    print(f"  Selected Tool: {trace.dispatched_tool.upper()}")
    print(f"  Calibrated Confidence: {trace.confidence:.4f} (Threshold: {trace.confidence_threshold:.2f})")
    print(f"  Confidence Passed: {trace.confidence_passed}")
    print(f"  Decider Latency: {trace.decider_latency_ms:.2f} ms")
    print(f"  Probability Distribution: {json.dumps({k: round(v, 4) for k, v in trace.probabilities.items()})}")
    print(f"  Tool Executed: '{result.tool_output.title}' (Success: {result.tool_output.success})")
    print(f"  Bedrock Live Verified: {not result.bedrock_response.is_simulated}")
    print(f"  Bedrock Latency: {trace.bedrock_latency_ms:.2f} ms")
    print(f"  Total Pipeline Latency: {trace.total_latency_ms:.2f} ms")
    print("-" * 75)

    print("\nFinal Synthesized Output:")
    print(result.final_explanation)
    print("\n" + "=" * 75)
    return True


if __name__ == "__main__":
    success = run_live_verification()
    sys.exit(0 if success else 1)
