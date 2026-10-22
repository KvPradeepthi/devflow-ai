"""Developer tools for DevFlow AI.

Provides four specialized tools:
1. explain_code: Analyzes code structure, syntax, and programming concepts.
2. debug_error: Analyzes exceptions, stack traces, and runtime errors, distinguishing facts from hypotheses.
3. analyze_test: Interprets supplied test failures, assertion errors, and test runner logs.
4. search_docs: Retrieves relevant technical documentation from local markdown files.

Safety notice: These tools perform safe static analysis and retrieval; they do NOT
execute arbitrary shell commands or untrusted code.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolOutput:
    """Standardized output produced by all DevFlow AI tools."""
    tool_name: str
    success: bool
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
        }


def explain_code(request: str, context: str = "") -> ToolOutput:
    """Analyze code or programming concepts and extract structural breakdown."""
    combined = (context.strip() or request.strip())
    if not combined:
        return ToolOutput(
            tool_name="explain_code",
            success=False,
            title="Missing Code Snippet",
            content="Please provide the code snippet or programming concept you would like explained.",
            metadata={"lines": 0, "detected_language": "unknown"},
        )

    # Detect programming language patterns
    detected_lang = "general"
    if re.search(r"\b(def|class|import|from|self|lambda)\b", combined):
        detected_lang = "python"
    elif re.search(r"\b(function|const|let|var|console\.log|=>)\b", combined):
        detected_lang = "javascript/typescript"
    elif re.search(r"\b(public|private|void|static\s+void\s+main)\b", combined):
        detected_lang = "java/c#"
    elif re.search(r"\b(fn|pub|impl|mut|let\s+mut)\b", combined):
        detected_lang = "rust"
    elif re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\b", combined, re.IGNORECASE):
        detected_lang = "sql"

    # Extract declared functions, classes, imports
    functions = re.findall(r"(?:def|function|fn)\s+([a-zA-Z0-9_]+)", combined)
    classes = re.findall(r"(?:class)\s+([a-zA-Z0-9_]+)", combined)
    imports = re.findall(r"(?:import\s+([a-zA-Z0-9_,\s]+)|from\s+([a-zA-Z0-9_\.]+)\s+import)", combined)

    lines = [line for line in combined.splitlines() if line.strip()]

    summary_parts = [
        f"**Language Detected:** {detected_lang.title()}",
        f"**Total Non-Empty Lines:** {len(lines)}",
    ]
    if classes:
        summary_parts.append(f"**Classes Identified:** {', '.join(classes)}")
    if functions:
        summary_parts.append(f"**Functions/Methods Identified:** {', '.join(functions)}")
    if imports:
        summary_parts.append(f"**External Modules Imported:** {len(imports)}")

    content = (
        "### Static Code Inspection\n"
        + "\n".join(f"- {p}" for p in summary_parts)
        + "\n\n### Code Content Prepared for Explanation\n```"
        + (detected_lang if "/" not in detected_lang else "text")
        + f"\n{combined}\n```"
    )

    return ToolOutput(
        tool_name="explain_code",
        success=True,
        title="Code Analysis & Structural Breakdown",
        content=content,
        metadata={
            "detected_language": detected_lang,
            "line_count": len(lines),
            "functions": functions,
            "classes": classes,
        },
    )


def debug_error(request: str, context: str = "") -> ToolOutput:
    """Analyze exceptions and stack traces, distinguishing verified facts from hypotheses."""
    combined = (context.strip() or request.strip())
    if not combined:
        return ToolOutput(
            tool_name="debug_error",
            success=False,
            title="Missing Error Details",
            content="Please provide the error message or stack trace to debug.",
            metadata={"frames_found": 0},
        )

    # Extract exception types and messages
    exception_matches = re.findall(r"([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception):\s*(.*)", combined)
    traceback_frames = re.findall(r'File "([^"]+)", line (\d+)(?:, in ([a-zA-Z0-9_]+))?', combined)

    # Distinguish verified facts vs hypotheses
    verified_facts = []
    if exception_matches:
        for exc_type, msg in exception_matches:
            verified_facts.append(f"Exception Type: `{exc_type}` with message: `{msg.strip()}`")
    else:
        # Fallback error detection
        generic_err = re.search(r"(?:error|exception|failed|fatal):\s*(.*)", combined, re.IGNORECASE)
        if generic_err:
            verified_facts.append(f"Observed Error Indicator: `{generic_err.group(0).strip()}`")
        else:
            verified_facts.append("No standard Python/Java stack trace pattern detected; inspecting raw log text.")

    if traceback_frames:
        for file_path, line_no, func_name in traceback_frames[-3:]:
            func_str = f" in function `{func_name}`" if func_name else ""
            verified_facts.append(f"Traceback frame: `{file_path}` at line `{line_no}`{func_str}")

    hypotheses = [
        "Check variable initialization and null/None checks prior to the reported line.",
        "Verify imported package versions and environment compatibility.",
        "Inspect potential type mismatches or unhandled edge cases in input parameters.",
    ]

    content_lines = [
        "### 1. Verified Trace Facts",
        *(f"- {fact}" for fact in verified_facts),
        "",
        "### 2. Hypotheses for Root Cause",
        *(f"- {hyp}" for hyp in hypotheses),
        "",
        "### 3. Recommended Safe Troubleshooting Steps",
        "- Reproduce the failure in an isolated test case with minimal inputs.",
        "- Add assertions or debug logging directly preceding the throwing line.",
        "- Verify file paths, permissions, and network endpoints if I/O is involved.",
    ]

    return ToolOutput(
        tool_name="debug_error",
        success=True,
        title="Error & Traceback Diagnostic",
        content="\n".join(content_lines),
        metadata={
            "exceptions_found": [m[0] for m in exception_matches],
            "frames_count": len(traceback_frames),
        },
    )


def analyze_test(request: str, context: str = "") -> ToolOutput:
    """Interpret supplied test failures, assertions, and test runner logs.

    Note: This tool performs static log inspection and does NOT execute any test suites.
    """
    combined = (context.strip() or request.strip())
    if not combined:
        return ToolOutput(
            tool_name="analyze_test",
            success=False,
            title="Missing Test Failure Output",
            content="Please provide the failing test output or assertion log to analyze.",
            metadata={"assertions_found": 0},
        )

    # Detect test runners
    runner = "generic test runner"
    lower_combined = combined.lower()
    if (
        "pytest" in lower_combined
        or "failures ===" in lower_combined
        or "failed in " in lower_combined
        or "failed " in combined
    ):
        runner = "pytest"
    elif "unittest" in lower_combined or "fail:" in lower_combined:
        runner = "unittest"
    elif "jest" in lower_combined or "fail " in lower_combined:
        runner = "jest"

    # Extract test names and assertions
    failed_tests = re.findall(r"(?:FAILED|FAIL:?)\s+([a-zA-Z0-9_\.:\-\/]+)", combined)
    assertions = re.findall(r"(?:assert|AssertionError:?)\s+(.*)", combined)
    expected_vs_actual = re.findall(r"(?:Expected|expected):\s*(.*?)\s*(?:Received|actual|got):\s*(.*)", combined, re.IGNORECASE)

    analysis_points = [
        f"**Test Framework Detected:** `{runner}`",
        "**Execution Status Note:** *This analysis was conducted purely on the provided log text; no tests were run by this tool.*",
    ]
    if failed_tests:
        analysis_points.append(f"**Failing Test Target(s):** {', '.join(f'`{t}`' for t in failed_tests)}")
    if assertions:
        analysis_points.append(f"**Assertion Expressions Identified:** {len(assertions)}")
        for a in assertions[:3]:
            analysis_points.append(f"  - `{a.strip()}`")
    if expected_vs_actual:
        for exp, act in expected_vs_actual[:2]:
            analysis_points.append(f"**Discrepancy:** Expected `{exp.strip()}` vs Received `{act.strip()}`")

    content_lines = [
        "### Test Failure Diagnostic Summary",
        *(f"- {p}" for p in analysis_points),
        "",
        "### Key Areas to Investigate",
        "- Verify whether recent code modifications altered return types or contract schemas.",
        "- Check fixture setup and teardown isolation for shared state leakage.",
        "- Validate floating point, timezone, or string whitespace differences.",
    ]

    return ToolOutput(
        tool_name="analyze_test",
        success=True,
        title="Test Failure Analysis",
        content="\n".join(content_lines),
        metadata={
            "runner": runner,
            "failed_tests": failed_tests,
            "assertion_count": len(assertions),
        },
    )


def search_docs(request: str, context: str = "", docs_dir: str | Path | None = None) -> ToolOutput:
    """Search local documentation collection using term-frequency matching with citations."""
    query = (request.strip() + " " + context.strip()).strip().lower()
    if not query:
        return ToolOutput(
            tool_name="search_docs",
            success=False,
            title="Empty Documentation Query",
            content="Please provide a search query or technical topic to search within the documentation.",
            metadata={"matches_found": 0},
        )

    if docs_dir is None:
        docs_dir = Path(__file__).parent / "docs_collection"
    else:
        docs_dir = Path(docs_dir)

    if not docs_dir.exists() or not docs_dir.is_dir():
        return ToolOutput(
            tool_name="search_docs",
            success=False,
            title="Documentation Collection Unavailable",
            content=f"Documentation repository directory `{docs_dir}` does not exist.",
            metadata={"matches_found": 0},
        )

    # Tokenize query, discarding common stop words
    stop_words = {"the", "a", "an", "is", "in", "it", "to", "for", "of", "and", "or", "how", "what", "can", "i"}
    query_tokens = [w for w in re.findall(r"\w+", query) if len(w) > 2 and w not in stop_words]

    if not query_tokens:
        query_tokens = [w for w in re.findall(r"\w+", query) if len(w) > 1]

    doc_files = list(docs_dir.glob("*.md"))
    scored_results = []

    for file_path in doc_files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue

        lower_content = content.lower()
        score = 0
        matches = []

        for token in query_tokens:
            count = lower_content.count(token)
            if count > 0:
                score += count
                matches.append(token)

        # Check for phrase matches
        for phrase in [request.strip().lower()]:
            if phrase and len(phrase) > 4 and phrase in lower_content:
                score += 10

        if score > 0:
            # Extract most relevant paragraph
            paragraphs = content.split("\n\n")
            best_para = ""
            best_para_score = -1
            for p in paragraphs:
                p_lower = p.lower()
                p_score = sum(p_lower.count(t) for t in query_tokens)
                if p_score > best_para_score:
                    best_para_score = p_score
                    best_para = p.strip()

            snippet = best_para[:350] + ("..." if len(best_para) > 350 else "")
            scored_results.append({
                "source": file_path.name,
                "score": score,
                "matched_tokens": list(set(matches)),
                "snippet": snippet,
            })

    scored_results.sort(key=lambda r: r["score"], reverse=True)

    if not scored_results:
        return ToolOutput(
            tool_name="search_docs",
            success=True,
            title="No Matching Documentation Found",
            content=f"No relevant documentation found in the local repository for query tokens: `{', '.join(query_tokens)}`.\n\nPlease refine your search query or verify that the topic is documented.",
            metadata={"matches_found": 0, "query_tokens": query_tokens},
        )

    # Format matching documentation with citations
    content_lines = [
        f"### Documentation Search Results ({len(scored_results)} sources found)",
        "",
    ]
    for idx, res in enumerate(scored_results[:3], 1):
        content_lines.extend([
            f"#### Source {idx}: `{res['source']}` (Relevance Score: {res['score']})",
            f"**Matched Keywords:** {', '.join(res['matched_tokens'])}",
            "",
            "> " + res["snippet"].replace("\n", "\n> "),
            "",
        ])

    return ToolOutput(
        tool_name="search_docs",
        success=True,
        title="Documentation Search Results",
        content="\n".join(content_lines),
        metadata={
            "matches_found": len(scored_results),
            "top_sources": [r["source"] for r in scored_results[:3]],
            "query_tokens": query_tokens,
        },
    )


# Registry of available tools
TOOL_REGISTRY = {
    "explain_code": explain_code,
    "debug_error": debug_error,
    "analyze_test": analyze_test,
    "search_docs": search_docs,
}
