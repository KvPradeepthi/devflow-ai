"""Standalone Reusable Decision Pattern Example: #tool-selection.

Demonstrates how to route developer tasks using Strands Decider 2B
independent of any UI framework.

Usage:
    python example.py
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from devflow.decider import StrandsDeciderClient, VALID_TOOLS
from devflow.tools import TOOL_REGISTRY


def run_pattern_demonstration():
    print("=" * 70)
    print("Strands Decider 2B Reusable Decision Pattern: #tool-selection")
    print("=" * 70)

    # Initialize Decider Client
    endpoint = os.environ.get("STRANDS_DECIDER_URL", "http://localhost:8099")
    decider = StrandsDeciderClient(endpoint_url=endpoint)

    # Health check
    health = decider.check_health()
    if health["online"]:
        print(f"[*] Decider Server: ONLINE at {endpoint}")
        print(f"[*] Model Checkpoint: {health.get('details', {}).get('checkpoint', 'unknown')}")
    else:
        print(f"[!] Decider Server: OFFLINE at {endpoint} ({health['error']})")
        print("[!] Running with local calibrated offline decision engine.")

    print("\nAllowed Tool Choices:")
    for tool_name, desc in VALID_TOOLS.items():
        print(f"  - {tool_name:<14}: {desc}")

    # Representative test queries
    test_cases = [
        {
            "category": "Code Explanation",
            "request": "Can you explain how this async generator works with the yield statement?",
            "context": "async def stream_items(n):\n    for i in range(n):\n        await asyncio.sleep(0.1)\n        yield i * 2",
        },
        {
            "category": "Error Debugging",
            "request": "Why am I getting this KeyError when parsing API response?",
            "context": 'Traceback (most recent call last):\n  File "app.py", line 42, in process_user\n    token = payload["auth"]["token"]\nKeyError: \'auth\'',
        },
        {
            "category": "Test Analysis",
            "request": "Our unit test is failing on assertion mismatch, what broke?",
            "context": "FAILED tests/test_math.py::test_calculate_tax - AssertionError: assert 105.0 == 100.0\n+  where 105.0 = calculate_tax(100.0, 0.05)",
        },
        {
            "category": "Documentation Search",
            "request": "How do I configure the Strands Decider System One HTTP server port?",
            "context": "Need reference for strands-decider serve arguments.",
        },
    ]

    print("\n" + "-" * 70)
    print("Evaluating Test Cases Through Decider 2B #tool-selection Pattern")
    print("-" * 70)

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[Case {idx}] Category: {tc['category']}")
        print(f"Request: \"{tc['request']}\"")
        if tc["context"]:
            first_line = tc["context"].splitlines()[0]
            print(f"Context snippet: {first_line[:65]}...")

        # 1. Ask Decider 2B
        decision = decider.select_tool(user_request=tc["request"], context=tc["context"])

        print(f"-> Selected Tool: {decision.selected_tool.upper()}")
        print(f"-> Calibrated Confidence: {decision.confidence:.4f}")
        print(f"-> Inference Latency: {decision.latency_ms:.2f} ms")
        print(f"-> Distribution: {json.dumps({k: round(v, 3) for k, v in decision.probabilities.items()})}")

        # 2. Dispatch Tool
        tool_func = TOOL_REGISTRY[decision.selected_tool]
        output = tool_func(tc["request"], tc["context"])
        print(f"-> Tool Executed: '{output.title}' (Success: {output.success})")

    print("\n" + "=" * 70)
    print("Pattern demonstration completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_pattern_demonstration()
