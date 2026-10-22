# Reusable Decision Pattern: #tool-selection with Strands Decider 2B

This pattern provides a drop-in, production-ready recipe for using **Strands Decider 2B** (`StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`) as a high-speed, zero-token-waste routing layer for agentic developer tools.

---

## 1. Pattern Overview

In traditional agent architectures, a heavy generative model (e.g., Claude 3.5 Sonnet, GPT-4o) is called simply to decide which tool to execute. This incurs 1,500ms to 4,000ms of latency, consumes hundreds of output tokens, and introduces stochastic failure modes into deterministic tool dispatching.

The **#tool-selection** pattern decouples intent routing from text generation:
- **System One (Fast / Calibrated)**: Strands Decider 2B evaluates the user request and context in parallel (typically 20ms–115ms), outputting a mathematical probability distribution and calibrated confidence score over candidate tools.
- **System Two (Generative / Grounded)**: Only after the tool executes and gathers factual context is Amazon Bedrock invoked to generate the user-facing explanation.

```
Developer Request + Context
            │
            ▼
┌───────────────────────────────┐
│     Strands Decider 2B        │ ◄── POST /v1/systemone (Port 8099)
│  (System One: #tool-selection) │     ChoiceQuestion evaluation
└───────────────┬───────────────┘
                │ Selected Choice + Calibrated Confidence
                ▼
┌───────────────────────────────┐
│       Confidence Policy       │ ── Low Conf (< 0.60) ──► Clarification / Fallback
└───────────────┬───────────────┘
                │ High Conf (>= 0.60)
                ▼
┌───────────────────────────────┐
│     Specialized Tool Run      │ (explain_code | debug_error | analyze_test | search_docs)
└───────────────┬───────────────┘
                │ Grounded Diagnostic Findings
                ▼
┌───────────────────────────────┐
│       Amazon Bedrock          │ ◄── Converse API (Claude 3 Haiku / Nova)
│   (System Two: Explanation)   │
└───────────────┬───────────────┘
                │
                ▼
      Final Verified Answer
```

---

## 2. Decision State Specification

The state is a normalized text string combining the developer's natural language request and optional context (code snippet, stack trace, test runner logs, or documentation keywords):

```python
state = f"{user_request.strip()}\n\nContext:\n{context.strip()}"
```

---

## 3. Typed Question Schema (Official Wire Format)

Strands Decider 2B accepts typed questions conforming to the official `ChoiceQuestion` specification:

```json
{
  "state": "Why am I getting KeyError: 'auth' at line 42?",
  "model": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
  "questions": {
    "tool_choice": {
      "type": "choice",
      "instructions": "Select the most appropriate developer tool to process the user's request.",
      "criteria": {
        "explain_code": "Explain code structure, syntax, logic, algorithms, or programming concepts.",
        "debug_error": "Analyze exceptions, stack traces, crash logs, and runtime error messages.",
        "analyze_test": "Interpret failing test output, assertions, test runner logs, and expected vs actual values.",
        "search_docs": "Retrieve relevant documentation, API specifications, and architectural guides from the local documentation collection."
      }
    }
  }
}
```

---

## 4. Permitted Choice Options & Tool Mapping

| Choice Option | Description | Target Function |
| :--- | :--- | :--- |
| `explain_code` | Code structure, syntax, logic, or algorithmic walkthroughs | `devflow.tools.explain_code` |
| `debug_error` | Exceptions, tracebacks, runtime crash logs | `devflow.tools.debug_error` |
| `analyze_test` | Test runner logs, assertion errors, expected vs actual mismatch | `devflow.tools.analyze_test` |
| `search_docs` | Architectural guides, SDK references, setup documentation | `devflow.tools.search_docs` |

---

## 5. Confidence Calculation & Policy Guardrails

### Official Confidence Formula
For a choice question with $N$ candidate options, Strands Decider calculates normalized confidence as:

$$\text{Confidence} = \max\left(0, \min\left(1, \frac{N \cdot p_{\max} - 1}{N - 1}\right)\right)$$

- If probabilities are uniform ($p_i = 1/N$), $\text{Confidence} = 0.0$.
- If one option has probability $1.0$, $\text{Confidence} = 1.0$.

### Policy Threshold
- **Default Threshold**: $0.60$
- **Confidence $\ge 0.60$**: Directly dispatch the selected tool.
- **Confidence $< 0.60$**: Request clarification from the developer or route to safe documentation search, recording the event in the decision trace.

---

## 6. How to Run the Example Independently

Run the standalone demonstration script:

```powershell
# Using the project virtual environment
E:\devflow_venv\Scripts\python.exe decision_patterns/tool_selection/example.py
```

Output:
```text
======================================================================
Strands Decider 2B Reusable Decision Pattern: #tool-selection
======================================================================
Allowed Tool Choices:
  - explain_code  : Explain code structure, syntax, logic...
  - debug_error   : Analyze exceptions, stack traces...
  - analyze_test  : Interpret failing test output...
  - search_docs   : Retrieve relevant documentation...

[Case 1] Category: Code Explanation
-> Selected Tool: EXPLAIN_CODE (Confidence: 0.9999, Latency: 28.54 ms)
[Case 2] Category: Error Debugging
-> Selected Tool: DEBUG_ERROR (Confidence: 1.0000, Latency: 28.57 ms)
[Case 3] Category: Test Analysis
-> Selected Tool: ANALYZE_TEST (Confidence: 0.9945, Latency: 28.56 ms)
[Case 4] Category: Documentation Search
-> Selected Tool: SEARCH_DOCS (Confidence: 1.0000, Latency: 28.53 ms)
```

---

## 7. Adapting This Pattern for Any Agent

To adapt this pattern to a new agent domain (e.g., DevOps incident triage, customer support, or database querying):

1. **Update `questions.json`**: Define your candidate choices and concrete criteria in the dictionary.
2. **Implement Handlers**: Write functions for each choice name matching the signature `handle_choice(state: str) -> Result`.
3. **Set Threshold**: Tune the confidence threshold in your router (typically $0.55$ to $0.70$).
4. **Deploy Decider 2B**: Start the official server via:
   ```bash
   strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
   ```
5. **Route Through `StrandsDeciderClient`**: Use the typed request format to query decisions in $<50\text{ms}$.
