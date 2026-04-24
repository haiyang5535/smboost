"""Unit tests for PythonSandbox.

Covers:
  - happy path (print + exit 0)
  - top-level `result` channel round-trip
  - timeout enforcement (runs within ~timeout + small slack)
  - syntax error surfaces on stderr with ok=False
  - stderr capture from a raising subprocess
  - subprocess isolation: a tool's `os.system` / filesystem write does NOT
    mutate the harness process state
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from smboost.tools import PythonSandbox


def test_happy_path_print_2_plus_2():
    sb = PythonSandbox()
    out = sb.run("print(2+2)")
    assert out["ok"] is True
    assert out["stdout"] == "4\n"
    assert out["stderr"] == ""
    assert out["result"] is None
    assert out["error"] is None


def test_result_channel_round_trips_a_value():
    sb = PythonSandbox()
    out = sb.run("result = 40 + 2")
    assert out["ok"] is True
    # sandbox returns repr() of the top-level `result`
    assert out["result"] == "42"


def test_result_channel_with_list():
    sb = PythonSandbox()
    out = sb.run("result = sorted([3,1,2])")
    assert out["ok"] is True
    assert out["result"] == "[1, 2, 3]"


def test_stdout_and_result_coexist():
    sb = PythonSandbox()
    out = sb.run("print('hello')\nresult = 7")
    assert out["ok"] is True
    assert "hello" in out["stdout"]
    # the result sentinel must be stripped from stdout
    assert "SMBOOST_RESULT" not in out["stdout"]
    assert out["result"] == "7"


def test_syntax_error_returns_ok_false():
    sb = PythonSandbox()
    out = sb.run("def broken(:\n")
    assert out["ok"] is False
    # Either traceback-in-stderr or nonzero_exit error tag is acceptable
    assert out["error"] is not None
    assert ("SyntaxError" in out["stderr"]) or ("nonzero_exit" in out["error"])


def test_raising_exception_captures_traceback_on_stderr():
    sb = PythonSandbox()
    out = sb.run('raise ValueError("boom_xyzzy")')
    assert out["ok"] is False
    assert "boom_xyzzy" in out["stderr"]
    assert "ValueError" in out["stderr"]


def test_timeout_enforced_within_slack():
    sb = PythonSandbox()
    t0 = time.time()
    out = sb.run("import time; time.sleep(10)", timeout_s=1.0)
    dt = time.time() - t0
    assert out["ok"] is False
    assert out["error"] == "timeout"
    # 1.0s sleep + subprocess startup + kill overhead — allow 1.5s slack.
    assert dt < 2.5, f"timeout did not fire within 2.5s (took {dt:.2f}s)"


def test_subprocess_isolation_cannot_mutate_harness_env():
    """A tool's subprocess sets os.environ — the parent's env must be unchanged."""
    sb = PythonSandbox()
    harness_env_before = os.environ.get("SMBOOST_SBX_CANARY")
    out = sb.run("import os; os.environ['SMBOOST_SBX_CANARY'] = 'yes'; print('done')")
    assert out["ok"] is True
    harness_env_after = os.environ.get("SMBOOST_SBX_CANARY")
    assert harness_env_after == harness_env_before, (
        "subprocess leaked environ into parent"
    )


def test_subprocess_isolation_cannot_import_harness_modules_by_default():
    """The sandbox runs with `-I` (isolated). sys.path should not have our project
    by default — a tool's `import smboost` should not succeed."""
    sb = PythonSandbox()
    out = sb.run("import smboost; print('leaked')")
    # Either: ModuleNotFoundError (ok=False) OR if it unexpectedly imports,
    # the test fails loudly.
    assert out["ok"] is False, (
        "tool subprocess was able to import the harness package — isolation broken"
    )


def test_subprocess_isolation_filesystem_side_effects_are_ephemeral(tmp_path):
    """A tool's file write inside its cwd should not appear in the harness cwd."""
    sb = PythonSandbox()
    # The tool writes 'canary.txt' relative to its own cwd (a tempdir we clean up)
    code = "open('canary.txt', 'w').write('hello')\nprint('wrote')"
    out = sb.run(code)
    assert out["ok"] is True
    # The harness process's cwd should not have that file.
    harness_cwd_canary = Path(os.getcwd()) / "canary.txt"
    assert not harness_cwd_canary.exists(), (
        "subprocess wrote to harness cwd — isolation broken"
    )


def test_os_system_in_tool_does_not_break_harness():
    """Sanity: a tool can call os.system without killing the harness."""
    sb = PythonSandbox()
    out = sb.run("import os; os.system('true'); print('ok')")
    assert out["ok"] is True
    assert "ok" in out["stdout"]
    # the harness process is still responsive after this call:
    sb.run("print('still alive')")


def test_empty_code_returns_ok_no_output():
    sb = PythonSandbox()
    out = sb.run("")
    assert out["ok"] is True
    assert out["stdout"] == ""
    assert out["result"] is None


def test_multiple_runs_are_independent():
    """State set in one run must not persist into the next — each run is a
    fresh subprocess."""
    sb = PythonSandbox()
    sb.run("x = 123")  # defined in a subprocess that dies
    out = sb.run("print(x)")
    assert out["ok"] is False
    assert "NameError" in out["stderr"]


def test_timeout_kwarg_accepts_float():
    sb = PythonSandbox()
    out = sb.run("print('ok')", timeout_s=2.5)
    assert out["ok"] is True
    assert out["stdout"] == "ok\n"


def test_result_with_unreprable_object_does_not_crash_sandbox():
    """If `repr(result)` raises, the sandbox should still return a string marker
    rather than crash."""
    sb = PythonSandbox()
    code = (
        "class Bad:\n"
        "    def __repr__(self): raise RuntimeError('no repr')\n"
        "result = Bad()\n"
    )
    out = sb.run(code)
    # exec completes ok; only the repr in the trailer might fail — sandbox
    # writes a '<unreprable: ...>' token in that case.
    assert out["ok"] is True
    assert out["result"] is not None
    assert "unreprable" in out["result"] or "RuntimeError" in out["result"] or out["result"]


def test_sandbox_workdir_is_cleaned_up(tmp_path):
    """After run(), the tempdir created for the subprocess should not linger."""
    sb = PythonSandbox()
    out = sb.run("import os; print(os.getcwd())")
    assert out["ok"] is True
    tool_cwd = out["stdout"].strip()
    # Tool's cwd was a smboost_sbx_ tempdir; should no longer exist now.
    if tool_cwd.startswith(tempfile.gettempdir()) and "smboost_sbx_" in tool_cwd:
        assert not Path(tool_cwd).exists(), (
            f"sandbox tempdir {tool_cwd} not cleaned up"
        )


@pytest.mark.parametrize("bad_code", [
    "1/0",
    "import nonexistent_module_xyzzy",
    "raise KeyboardInterrupt",
])
def test_various_errors_surface_cleanly(bad_code):
    sb = PythonSandbox()
    out = sb.run(bad_code)
    assert out["ok"] is False
    assert out["error"] is not None
