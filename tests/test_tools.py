"""Unit tests for DevFlow AI developer tools."""

from __future__ import annotations

import pytest
from devflow.tools import (
    analyze_test,
    debug_error,
    explain_code,
    search_docs,
    TOOL_REGISTRY,
)


class TestExplainCodeTool:
    def test_explain_python_code(self):
        snippet = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
        res = explain_code("Explain this recursive function", snippet)
        assert res.success is True
        assert res.tool_name == "explain_code"
        assert res.metadata["detected_language"] == "python"
        assert "fibonacci" in res.metadata["functions"]
        assert "fibonacci" in res.content

    def test_explain_empty_code(self):
        res = explain_code("", "")
        assert res.success is False
        assert "Missing Code Snippet" in res.title


class TestDebugErrorTool:
    def test_debug_keyerror_traceback(self):
        traceback_text = """
Traceback (most recent call last):
  File "server.py", line 85, in handle_request
    user_id = session["user"]["id"]
KeyError: 'user'
"""
        res = debug_error("Why did this fail?", traceback_text)
        assert res.success is True
        assert res.tool_name == "debug_error"
        assert "KeyError" in res.metadata["exceptions_found"]
        assert "server.py" in res.content
        assert "85" in res.content
        assert "Verified Trace Facts" in res.content

    def test_debug_empty_input(self):
        res = debug_error("", "")
        assert res.success is False
        assert "Missing Error Details" in res.title


class TestAnalyzeTestTool:
    def test_analyze_pytest_failure(self):
        pytest_log = """
=================================== FAILURES ===================================
_________________________________ test_pricing _________________________________
    def test_pricing():
>       assert calculate_total(100) == 120
E       AssertionError: assert 110 == 120

=========================== 1 failed in 0.12s ===========================
"""
        res = analyze_test("Why is this test failing?", pytest_log)
        assert res.success is True
        assert res.tool_name == "analyze_test"
        assert res.metadata["runner"] == "pytest"
        assert "AssertionError" in res.content or "assert" in res.content
        assert "no tests were run by this tool" in res.content

    def test_analyze_test_empty_input(self):
        res = analyze_test("", "")
        assert res.success is False
        assert "Missing Test Failure Output" in res.title


class TestSearchDocsTool:
    def test_search_docs_matching_query(self, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "strands.md").write_text(
            "# Strands Decider\nStrands Decider 2B is a pointer readout model.",
            encoding="utf-8",
        )

        res = search_docs("How does Strands Decider work?", "", docs_dir=doc_dir)
        assert res.success is True
        assert res.tool_name == "search_docs"
        assert res.metadata["matches_found"] > 0
        assert "strands.md" in res.metadata["top_sources"]

    def test_search_docs_no_match(self, tmp_path):
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "python.md").write_text("Python formatting guidelines.", encoding="utf-8")

        res = search_docs("quantum entanglement teleportation astrophysics", "", docs_dir=doc_dir)
        assert res.success is True
        assert res.metadata["matches_found"] == 0
        assert "No relevant documentation found" in res.content

    def test_search_docs_empty_query(self):
        res = search_docs("", "")
        assert res.success is False
        assert "Empty Documentation Query" in res.title


class TestUntrustedInputSafety:
    def test_untrusted_shell_injection_attempt(self):
        malicious = "explain this: ; rm -rf / ; cat /etc/passwd && echo pwned"
        res = explain_code(malicious, "")
        # Tool safely treats it as text/code, never executes it
        assert res.success is True
        assert "rm -rf" in res.content
        assert res.tool_name == "explain_code"
