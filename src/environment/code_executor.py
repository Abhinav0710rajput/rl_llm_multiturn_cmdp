"""
Sandboxed Python code executor.
Runs agent-generated code against a problem's test cases and returns pass@1.
"""

import resource
import subprocess
import sys
import tempfile
import textwrap
from typing import List, Dict


# ── Test case → assert statement conversion ───────────────────────────────────

def build_test_program(code: str, test_cases: List[Dict], entry_point: str) -> str:
    """
    Build a self-contained Python program from the agent's code + test cases.

    Each test case has:
        input:    string representation of argument(s), e.g. "[1,2,3], 2"
        output:   string representation of expected return, e.g. "[2,3,4]"
        relation: "==" or custom expression like "abs(candidate(x) - 0.5) < 1e-6"
    """
    lines = [textwrap.dedent(code), ""]

    # If the agent defined 'candidate' but tests use a different entry_point, alias it
    if entry_point != "candidate" and "def candidate(" in code:
        lines.append(f"{entry_point} = candidate")
        lines.append("")
    # If the agent used the original entry_point but tests reference 'candidate'
    elif entry_point == "candidate":
        pass  # no alias needed
    # If the agent used the entry_point name directly, no alias needed
    elif f"def {entry_point}(" in code:
        pass

    for tc in test_cases:
        inp = tc.get("input", "")
        out = tc.get("output", "")
        rel = tc.get("relation", "==")

        if rel == "==":
            # Standard equality check
            lines.append(f"assert {entry_point}({inp}) == {out}")
        elif "candidate" in rel:
            # Custom relation referencing 'candidate' — substitute with entry_point
            expr = rel.replace("candidate", entry_point)
            lines.append(f"assert {expr}")
        else:
            # Raw relation string — treat as full assert expression
            lines.append(f"assert {rel}")

    return "\n".join(lines)


def _build_single_test(code: str, test_case: Dict, entry_point: str) -> str:
    """Build a program that runs exactly one test case."""
    inp = test_case.get("input", "")
    out = test_case.get("output", "")
    rel = test_case.get("relation", "==")

    lines = [textwrap.dedent(code), ""]

    # Alias candidate -> entry_point if needed
    if entry_point != "candidate" and "def candidate(" in code:
        lines.append(f"{entry_point} = candidate")
        lines.append("")

    if rel == "==":
        lines.append(f"assert {entry_point}({inp}) == {out}")
    elif "candidate" in rel:
        expr = rel.replace("candidate", entry_point)
        lines.append(f"assert {expr}")
    else:
        lines.append(f"assert {rel}")

    return "\n".join(lines)


# ── Executor ──────────────────────────────────────────────────────────────────

class CodeExecutor:
    def __init__(self, cfg):
        self.timeout = cfg.code_executor.timeout
        self.partial_credit = cfg.code_executor.partial_credit

    def run(self, code: str, test_cases: List[Dict], entry_point: str) -> float:
        """
        Execute agent code against the test suite.

        Returns:
            pass@1 score in [0.0, 1.0]
            - 1.0  → all tests pass
            - 0.0  → execution error or all tests fail
            - 0.x  → fraction passing (only if partial_credit=True)
        """
        if not code.strip():
            return 0.0
        if not test_cases:
            return 0.0

        if self.partial_credit:
            return self._run_partial(code, test_cases, entry_point)
        else:
            return self._run_all(code, test_cases, entry_point)

    def _run_all(self, code: str, test_cases: List[Dict], entry_point: str) -> float:
        """Run all tests together. Returns 1.0 if all pass, 0.0 otherwise."""
        program = build_test_program(code, test_cases, entry_point)
        success = _execute_program(program, self.timeout)
        return 1.0 if success else 0.0

    def _run_partial(self, code: str, test_cases: List[Dict], entry_point: str) -> float:
        """Run each test case individually. Returns fraction that pass."""
        if not test_cases:
            return 0.0

        passed = 0
        for tc in test_cases:
            program = _build_single_test(code, tc, entry_point)
            if _execute_program(program, self.timeout):
                passed += 1

        return passed / len(test_cases)


def _execute_program(program: str, timeout: float) -> bool:
    """
    Run a Python program in a subprocess with timeout.
    Returns True if it exits with code 0, False otherwise.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(program)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Code extraction ───────────────────────────────────────────────────────────

def extract_code(action_text: str) -> str:
    """
    Extract Python code from the agent's [ANSWER] response.
    Handles:
      - Raw code after [ANSWER] tag
      - Markdown code blocks (```python ... ```)
    """
    # Strip [ANSWER] prefix
    text = action_text
    if text.upper().startswith("[ANSWER]"):
        text = text[len("[ANSWER]"):].strip()

    # Extract from markdown code block if present
    if "```" in text:
        start = text.find("```")
        # Skip language identifier line (```python)
        newline = text.find("\n", start)
        end = text.find("```", newline)
        if newline != -1 and end != -1:
            return text[newline + 1: end].strip()

    return text.strip()
