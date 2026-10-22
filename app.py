"""DevFlow AI: Developer Request Routing with Strands Decider 2B & Amazon Bedrock.

Streamlit interactive user interface.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure project root is available in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from devflow.bedrock_client import BedrockClient
from devflow.decider import StrandsDeciderClient, VALID_TOOLS
from devflow.router import DevFlowRouter, DevFlowResult

st.set_page_config(
    page_title="DevFlow AI — Strands Decider 2B + Amazon Bedrock",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
    }
    .tool-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .pill-explain { background-color: #DBEAFE; color: #1E40AF; }
    .pill-debug { background-color: #FEE2E2; color: #991B1B; }
    .pill-test { background-color: #FEF3C7; color: #92400E; }
    .pill-docs { background-color: #D1FAE5; color: #065F46; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Preset Examples
PRESET_EXAMPLES = {
    "Explain Code (Async Generator)": {
        "request": "Can you explain how this async generator works with the yield statement?",
        "context": "async def stream_items(n: int):\n    for i in range(n):\n        await asyncio.sleep(0.1)\n        yield i * 2",
    },
    "Debug Error (KeyError Traceback)": {
        "request": "Why am I getting this KeyError when parsing the response dictionary?",
        "context": 'Traceback (most recent call last):\n  File "app.py", line 42, in process_user\n    token = payload["auth"]["token"]\nKeyError: \'auth\'',
    },
    "Analyze Test (Pytest Assertion Failure)": {
        "request": "Our unit test failed on the discount assertion, what caused this discrepancy?",
        "context": "FAILED tests/test_billing.py::test_discount - AssertionError: assert 90.0 == 95.0\n+  where 90.0 = apply_discount(100.0, 0.10)\n-  where 95.0 = expected_value",
    },
    "Search Docs (Strands Decider Setup)": {
        "request": "How do I configure and serve Strands Decider 2B using the CLI on port 8099?",
        "context": "Need parameters for strands-decider serve and supported question types.",
    },
    "Ambiguous Query (Low-Confidence Gate)": {
        "request": "Can you check on things?",
        "context": "",
    },
}

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Strands Decider 2B")
    decider_endpoint = st.text_input(
        "Decider URL",
        value=os.environ.get("STRANDS_DECIDER_URL", "http://localhost:8099"),
        help="Endpoint of the strands-decider serve instance.",
    )

    decider_api_key = st.text_input(
        "API Key / Token (Optional)",
        value=os.environ.get("STRANDS_DECIDER_API_KEY", ""),
        type="password",
        help="Optional Bearer token if connecting to an authenticated remote Decider server.",
    )

    # Health check
    decider_client = StrandsDeciderClient(
        endpoint_url=decider_endpoint,
        api_key=decider_api_key,
    )
    health = decider_client.check_health()
    if health["online"]:
        st.success("🟢 Strands Decider 2B: Online (Live Inference Active)")
    else:
        st.error("🔴 Strands Decider 2B: Offline (Live Inference Unverified)")
        st.caption(f"⚠️ Server status: {health['error'][:80] if health.get('error') else 'offline'}")

    require_live_server = st.checkbox(
        "Strict Live Mode (Disable Fallback)",
        value=False,
        help="When checked, errors if the live Decider server is unreachable, ensuring no simulation is used.",
    )

    st.subheader("Confidence Policy")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="Decisions with confidence below this threshold trigger clarification or safe fallback.",
    )

    st.subheader("Amazon Bedrock")
    aws_region = st.selectbox(
        "AWS Region",
        ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        index=0,
    )
    bedrock_model = st.selectbox(
        "Bedrock Model ID",
        [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "amazon.nova-micro-v1:0",
            "amazon.titan-text-express-v1",
        ],
        index=0,
    )

    bedrock_client = BedrockClient(region=aws_region, model_id=bedrock_model)
    has_creds, creds_msg = bedrock_client.check_credentials()
    if has_creds:
        st.success(f"🟢 AWS Credentials: Active ({aws_region})")
    else:
        st.info("ℹ️ AWS Credentials not found; using grounded offline simulation.")

    st.markdown("---")
    st.markdown(
        "**Hackathon Deliverable Tags:**\n"
        "`#strands-decider-2b` `#tool-selection`"
    )

# Main View
st.markdown('<div class="main-title">⚡ DevFlow AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent Developer Request Routing with <b>Strands Decider 2B</b> & <b>Amazon Bedrock</b></div>',
    unsafe_allow_html=True,
)

# Preset Buttons
st.markdown("**Try a Sample Developer Request:**")
preset_cols = st.columns(len(PRESET_EXAMPLES))
selected_preset = None

for col, (label, data) in zip(preset_cols, PRESET_EXAMPLES.items()):
    short_label = label.split(" (")[0]
    if col.button(short_label, use_container_width=True):
        selected_preset = data

# Session State Initialization
if "user_request" not in st.session_state:
    st.session_state.user_request = (
        "Why am I getting this KeyError when parsing API response?"
    )
if "user_context" not in st.session_state:
    st.session_state.user_context = (
        'Traceback (most recent call last):\n  File "app.py", line 42, in process_user\n    token = payload["auth"]["token"]\nKeyError: \'auth\''
    )

if selected_preset:
    st.session_state.user_request = selected_preset["request"]
    st.session_state.user_context = selected_preset["context"]

# Input Form
with st.form("request_form"):
    req_input = st.text_input(
        "Developer Request:",
        value=st.session_state.user_request,
        placeholder="e.g., Explain this async function, debug this traceback, or search documentation...",
    )
    context_input = st.text_area(
        "Optional Code / Traceback / Test Log / Topic Context:",
        value=st.session_state.user_context,
        height=140,
        placeholder="Paste code snippet, stack trace, or test runner log here...",
    )
    submitted = st.form_submit_button("🚀 Route & Process Request", type="primary")

if submitted or "last_result" in st.session_state:
    if submitted:
        router = DevFlowRouter(
            decider_client=decider_client,
            bedrock_client=bedrock_client,
            confidence_threshold=confidence_threshold,
        )
        with st.spinner("Decider 2B evaluating request and Bedrock synthesizing response..."):
            result: DevFlowResult = router.route_and_execute(
                user_request=req_input,
                context=context_input,
                allow_offline_fallback=not require_live_server,
            )
            st.session_state.last_result = result
    else:
        result = st.session_state.last_result

    trace = result.trace
    tool_out = result.tool_output
    bedrock_res = result.bedrock_response

    # Prominent Audit Verification Banner
    if not trace.decider_is_live:
        st.warning(
            "⚠️ **OFFLINE DEMO MODE — LIVE DECIDER INFERENCE UNVERIFIED**  \n"
            f"The Strands Decider 2B server is offline or unreachable at `{decider_endpoint}`. "
            "This decision was produced by an **offline heuristic fallback simulation**.\n\n"
            "👉 **Hackathon Submission Requirement:** A simulated decision does *not* satisfy the official live Decider 2B requirement. "
            "To verify live inference, launch the model server via:  \n"
            "`strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099`"
        )
    else:
        st.success(
            "✅ **LIVE STRANDS DECIDER 2B INFERENCE VERIFIED**  \n"
            f"Decision successfully evaluated by live model `{trace.decider_model}` in `{trace.decider_latency_ms:.1f}ms` over official System One API."
        )

    # Top Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Selected Tool",
            value=trace.dispatched_tool.replace("_", " ").title(),
            delta="Confidence: " + f"{trace.confidence:.2%}",
        )
    with m2:
        st.metric(
            label="Confidence Gate",
            value="PASSED" if trace.confidence_passed else "LOW CONFIDENCE",
            delta=f"Threshold: {trace.confidence_threshold:.2f}",
        )
    with m3:
        st.metric(
            label="Decider Latency",
            value=f"{trace.decider_latency_ms:.1f} ms",
            delta="Live 2B Verified" if trace.decider_is_live else "Offline Simulation",
        )
    with m4:
        st.metric(
            label="Total Latency",
            value=f"{trace.total_latency_ms:.1f} ms",
            delta=f"Bedrock: {trace.bedrock_latency_ms:.0f} ms",
        )

    # Tool Selection Probability Distribution
    st.markdown("### 📊 Strands Decider 2B Probability Distribution")
    prob_cols = st.columns(len(trace.probabilities))
    for col, (tool, prob) in zip(prob_cols, trace.probabilities.items()):
        is_sel = tool == trace.dispatched_tool
        col.markdown(f"**{tool}**{' ⭐' if is_sel else ''}")
        col.progress(float(prob))
        col.caption(f"{prob:.1%}")

    # Results Layout
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown(f"### 🛠️ Tool Diagnostic Output: `{trace.dispatched_tool}`")
        if tool_out.success:
            st.success(f"Status: {tool_out.title}")
        else:
            st.warning(f"Status: {tool_out.title}")
        st.markdown(tool_out.content)

    with res_col2:
        st.markdown(f"### 🤖 Amazon Bedrock Synthesis")
        if bedrock_res.is_simulated:
            st.info(f"ℹ️ Model: `{trace.bedrock_model}` (Offline Simulation — AWS credentials not configured)")
        else:
            st.success(f"✅ Model: `{trace.bedrock_model}` (Live Bedrock Inference Verified)")
        st.markdown(result.final_explanation)

    # Expandable Auditable Decision Trace
    st.markdown("---")
    with st.expander("🔍 Auditable Decision Trace & Metadata (JSON)", expanded=False):
        st.json(result.to_dict())
