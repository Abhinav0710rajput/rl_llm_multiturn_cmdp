"""
Sandboxed Python code executor.
Runs agent-generated code against a problem's test cases and returns pass@1.
"""

import re
import resource
import subprocess
import sys
import tempfile
import textwrap
from typing import List, Dict


def _rename_function(code: str, entry_point: str) -> str:
    """
    Rename the first function definition in the code to entry_point.
    This makes the function name irrelevant — whatever the agent called it,
    the tests will find it under the expected name.
    """
    # Match 'def some_name(' and replace with 'def entry_point('
    return re.sub(r'def\s+\w+\s*\(', f'def {entry_point}(', code, count=1)


def _format_output(out) -> str:
    """
    Format a test case output value for insertion into an assert statement.
    Numeric and list outputs can be inserted directly; string outputs need quoting.
    """
    # If it's already a valid Python literal (number, list, bool, None), use as-is
    if isinstance(out, (int, float, bool, list, dict, type(None))):
        return repr(out)
    # String output — try to parse as a Python literal first
    s = str(out)
    try:
        import ast
        ast.literal_eval(s)
        # It's already a valid Python expression (e.g., "[1,2,3]", "True", "0.5")
        return s
    except (ValueError, SyntaxError):
        # It's a bare string (e.g., "fdcb") — needs quoting
        return repr(s)


def _expand_template_relation(rel: str, inp, entry_point: str) -> str:
    """
    Expand template-based test relations that use $demo$ and $input$ placeholders.

    Example relation:
        from $demo$ import poly\nimport math\nsolution = find_zero($input$)\nprint(math.fabs(poly($input$, solution)) < 1e-4)

    - $demo$ imports are stripped (helper functions like poly are already in the code)
    - $input$ is replaced with the actual input value
    - candidate is replaced with entry_point
    - print(expr) is converted to assert expr
    """
    inp_str = str(inp)

    # Replace template variables
    expanded = rel.replace("$input$", inp_str)
    expanded = expanded.replace("candidate", entry_point)

    # Remove $demo$ import lines (the functions are already defined in the code)
    result_lines = []
    for line in expanded.split("\n"):
        if "$demo$" in line:
            continue  # skip import from $demo$
        # Convert print(condition) to assert condition
        if line.strip().startswith("print(") and line.strip().endswith(")"):
            inner = line.strip()[6:-1]  # extract inside of print(...)
            result_lines.append(f"assert {inner}")
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


# ── Test case → assert statement conversion ───────────────────────────────────

def build_test_program(code: str, test_cases: List[Dict], entry_point: str) -> str:
    """
    Build a self-contained Python program from the agent's code + test cases.

    Each test case has:
        input:    string representation of argument(s), e.g. "[1,2,3], 2"
        output:   string representation of expected return, e.g. "[2,3,4]"
        relation: "==" or custom expression like "abs(candidate(x) - 0.5) < 1e-6"
    """
    # Rename the agent's function to match entry_point, regardless of what the agent called it
    code = _rename_function(code, entry_point)
    lines = [textwrap.dedent(code), ""]

    for tc in test_cases:
        inp = tc.get("input", "")
        out = tc.get("output", "")
        rel = tc.get("relation", "==")

        if rel == "==":
            # Standard equality check
            out_expr = _format_output(out)
            lines.append(f"assert {entry_point}({inp}) == {out_expr}")
        elif "$demo$" in rel or "$input$" in rel:
            # Template-based relation (e.g., find_zero uses poly from the same file)
            expr = _expand_template_relation(rel, inp, entry_point)
            lines.append(expr)
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

    code = _rename_function(code, entry_point)
    lines = [textwrap.dedent(code), ""]

    if rel == "==":
        out_expr = _format_output(out)
        lines.append(f"assert {entry_point}({inp}) == {out_expr}")
    elif "$demo$" in rel or "$input$" in rel:
        expr = _expand_template_relation(rel, inp, entry_point)
        lines.append(expr)
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
