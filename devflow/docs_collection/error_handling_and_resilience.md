# Error Handling and Resilience in Agent Systems

## The Two-Tier Architecture Pattern
Reliable AI agents combine fast deterministic filters (System One) with flexible generative engines (System Two):
- **Deterministic Routing**: Small, calibrated models like Strands Decider evaluate routing choices in milliseconds, preventing runaway LLM billing and reducing latency.
- **Confidence Gates**: Decisions with low confidence (< 0.60) trigger explicit clarification or safe fallback workflows rather than executing uncertain tools.

## Circuit Breakers and Timeouts
- Set strict client-side timeouts (e.g., 3.0s for routing, 15.0s for Bedrock generation).
- Distinguish between connection failure, HTTP status errors, and validation errors.
- Never silently swap in a hardcoded decision while claiming it came from a model.
