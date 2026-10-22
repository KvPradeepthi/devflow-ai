# Strands Decider 2B Architecture & Integration Guide

## Overview
Strands Decider 2B is a specialized non-generative decision model designed for agentic workflows. Instead of producing free-form natural language tokens, Decider 2B acts as a high-speed "System One" classifier that evaluates a given state and outputs calibrated confidence distributions over predefined options.

## Core Characteristics
- **Base Architecture**: Qwen3.5-2B-Base torso with a pointer/readout head via rank-16 LoRA adapter (`StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`).
- **Inference Speed**: Typical forward-pass latency is 20ms to 120ms, significantly faster than multi-second generative LLMs.
- **Calibrated Confidence**: Confidence scores are normalized mathematically:
  $$\text{Confidence} = \frac{N \cdot p_{\max} - 1}{N - 1}$$
  This ensures confidence is invariant to option count $N$ (0.0 for uniform distribution, 1.0 for absolute certainty).

## Question Primitives
Strands Decider provides three foundational question primitives:
1. **Choice Question (`choice`)**: Discrete categorical selection over $N$ named candidate options with criteria descriptions.
2. **Noul Question (`noul`)**: Calibrated binary yes/no verification returning true probability $P(\text{true})$.
3. **Score Question (`score`)**: Ordinal evaluation along an ascending numeric rubric (2 to 10 levels) returning expected score and variance-derived confidence.

## HTTP System One Interface
When deployed via `strands-decider serve <model-id> --port 8099`, Decider exposes two core endpoints:
- `GET /health`: Returns engine status, loaded checkpoint, slots, and device information.
- `POST /v1/systemone`: Evaluates a state against structured questions.
