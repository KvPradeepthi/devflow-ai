# DevFlow AI: Demonstration Video Script

**Target Duration:** 2 minutes 40 seconds (under the 3-minute hackathon maximum)  
**Hackathon:** AWS Agent Experience Hackathon: Strands Decider  
**Tags:** `#strands-decider-2b` `#tool-selection`  

---

## Storyboard & Timing Overview

| Scene | Time Window | Visual Focus | Audio / Spoken Narration |
| :---: | :---: | :--- | :--- |
| **1** | 0:00 – 0:25 | Title slide & Streamlit UI overview | Project concept, problem of LLM routing latency, two-tier architecture introduction. |
| **2** | 0:25 – 0:55 | Clicking Preset: "Debug Error" | Entering error traceback; live Decider 2B inference in 28ms; probability distribution. |
| **3** | 0:55 – 1:25 | Tool Diagnostic & Bedrock Synthesis | Tool output showing verified facts vs hypotheses; Amazon Bedrock generating concise grounded fix. |
| **4** | 1:25 – 1:55 | Ambiguous Query (Confidence Gate) | Entering vague query; confidence dropping below 0.60; policy guardrail triggering safe fallback. |
| **5** | 1:55 – 2:20 | Auditable Decision Trace (JSON) | Inspecting full audit trace: timestamps, latencies, probabilities, model parameters. |
| **6** | 2:20 – 2:40 | Reusable Pattern & Repo Wrap-Up | Showing `decision_patterns/tool_selection/`, CLI tests, GitHub repo, and hackathon tags. |

---

## Detailed Scene-by-Scene Script

### Scene 1: Introduction & The Routing Bottleneck (0:00 – 0:25)
- **Visual**: Screen recording opens on the DevFlow AI Streamlit application at `http://localhost:8501`. Show the clean interface with the Strands Decider 2B status badge and AWS Bedrock indicator in the sidebar.
- **Narrator**:
  > *"Welcome to DevFlow AI, an intelligent developer assistant built for the AWS Agent Experience Hackathon. Traditional AI agents use multi-billion parameter LLMs just to decide which tool to call—wasting up to three seconds and hundreds of tokens on basic routing. DevFlow AI solves this by introducing a Two-Tier Cognitive Architecture: Strands Decider 2B handles instant, zero-token System One routing, while Amazon Bedrock generates grounded, authoritative explanations."*

---

### Scene 2: Live Routing with Strands Decider 2B (0:25 – 0:55)
- **Visual**: Cursor clicks the preset button: **"Debug Error"**. The request field populates with:  
  `Why am I getting this KeyError when parsing API response?`  
  and the context area populates with a Python traceback showing `KeyError: 'auth'` at line 42.  
  Click **"Route & Process Request"**.
- **Narrator**:
  > *"Let's submit a real developer problem: a nested KeyError during JWT authentication. Behind the scenes, DevFlow AI sends a typed ChoiceQuestion directly to Strands Decider 2B over the System One API. Notice the metrics bar at the top: Decider 2B evaluated the request and context in just 28 milliseconds, selecting 'debug_error' with a calibrated confidence of 100%. Below, the probability distribution chart confirms that over 99% of probability mass was concentrated on the error debugging tool."*

---

### Scene 3: Tool Execution & Grounded Bedrock Generation (0:55 – 1:25)
- **Visual**: Scroll down to display the side-by-side results view.  
  Left column: **Tool Diagnostic Output (`debug_error`)** highlighting verified facts (file `app.py`, line 42, exception `KeyError`).  
  Right column: **Amazon Bedrock Synthesis** displaying a structured markdown resolution.
- **Narrator**:
  > *"Next, the selected tool executes safely, isolating verified facts from hypotheses. It identifies the exact throwing frame at line 42 without running any arbitrary shell commands. DevFlow AI then passes these structured findings to Amazon Bedrock using the Bedrock Converse API. Bedrock immediately synthesizes an authoritative code remediation, explaining how to use defensive `.get('auth', {})` access to prevent the crash."*

---

### Scene 4: The Confidence Gate in Action (1:25 – 1:55)
- **Visual**: Cursor clicks the preset button: **"Ambiguous Query"**.  
  Request: `Can you check on things?` Context: *(empty)*.  
  Click **"Route & Process Request"**.  
  Show the metrics bar displaying: `Confidence Gate: LOW CONFIDENCE` and confidence meter at `0.0%`.
- **Narrator**:
  > *"What happens when a developer provides an ambiguous request? Watch what happens when we submit 'Can you check on things?' without code or context. Strands Decider calculates a flat probability distribution, resulting in a calibrated confidence of zero. Because this falls below our configured 0.60 threshold, DevFlow AI's confidence guardrail triggers immediately, preventing ungrounded tool execution and requesting clarification."*

---

### Scene 5: Auditable Decision Trace (1:55 – 2:20)
- **Visual**: Click to expand the accordion: **"🔍 Auditable Decision Trace & Metadata (JSON)"**.  
  Highlight key fields: `request_id`, `decider.probabilities`, `decider.latency_ms`, `policy.confidence_passed`, `bedrock.latency_ms`.
- **Narrator**:
  > *"Every decision made by DevFlow AI is fully auditable. Expanding the decision trace reveals the complete telemetry: the exact Decider latency, probability distributions across all candidate tools, confidence gate status, and Bedrock token usage. This provides full observability for enterprise software teams."*

---

### Scene 6: Reusable Decision Pattern & Conclusion (2:20 – 2:40)
- **Visual**: Switch to VS Code terminal.  
  Show `decision_patterns/tool_selection/` with `questions.json` and run:  
  `python decision_patterns/tool_selection/example.py`  
  Show clean terminal output and display the GitHub repository page.
- **Narrator**:
  > *"We've packaged this entire workflow into a standalone, reusable decision pattern under `decision_patterns/tool_selection/`, complete with typed schemas, independent Python examples, and automated tests. DevFlow AI proves that separating fast System One decisions from generative synthesis creates faster, cheaper, and safer AI agents. Thank you for watching!"*

---

## Production Checklist for Recording
- [ ] Screen resolution set to 1920x1080 (1080p) at 60fps.
- [ ] Browser zoom set to 110% for crisp font readability.
- [ ] Synthetic data only (no real API keys or AWS credentials shown on screen).
- [ ] Total recording time strictly between 2 minutes 30 seconds and 2 minutes 45 seconds.
