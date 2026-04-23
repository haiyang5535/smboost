import pytest
import json
import base64
import zlib
import pickle
from benchmarks.livecodebench import sandbox
from benchmarks.livecodebench.sandbox import run, decode_test_cases


@pytest.fixture(autouse=True)
def _reset_pickle_skip_warned():
    """Reset the module-level one-shot warning flag between tests."""
    sandbox._pickle_skip_warned = False
    yield
    sandbox._pickle_skip_warned = False


def test_decode_test_cases(monkeypatch):
    # Pickled payloads are only decoded when the opt-in env var is set (F6).
    monkeypatch.setenv("SMBOOST_ALLOW_PICKLE_TEST_CASES", "1")
    cases = [{"input": "1", "output": "2", "testtype": "stdin"}]
    encoded = base64.b64encode(zlib.compress(pickle.dumps(cases))).decode('utf-8')
    test_code_str = json.dumps([encoded])

    decoded = decode_test_cases(test_code_str)
    assert len(decoded) == 1
    assert decoded[0]["input"] == "1"


def test_decode_test_cases_skips_pickle_without_env_var(monkeypatch, capsys):
    """By default (F6), pickled LCB entries are skipped and a warning is printed once."""
    monkeypatch.delenv("SMBOOST_ALLOW_PICKLE_TEST_CASES", raising=False)
    cases = [{"input": "1", "output": "2", "testtype": "stdin"}]
    encoded = base64.b64encode(zlib.compress(pickle.dumps(cases))).decode("utf-8")
    # One JSON-native case alongside the pickled one so we can verify we keep
    # the safe entries and drop only the pickled payload.
    native_case = {"input": "9", "output": "10", "testtype": "stdin"}
    test_code_str = json.dumps([encoded, native_case])

    decoded = decode_test_cases(test_code_str)

    # Only the native JSON case survives; the pickled one is silently dropped
    # (non-crashing) and a one-shot warning is printed to stderr.
    assert decoded == [native_case]
    captured = capsys.readouterr()
    assert "SMBOOST_ALLOW_PICKLE_TEST_CASES" in captured.err
    assert "skipping pickled" in captured.err


def test_decode_test_cases_allows_pickle_with_env_var(monkeypatch):
    """With SMBOOST_ALLOW_PICKLE_TEST_CASES=1, pickled entries are decoded (F6)."""
    monkeypatch.setenv("SMBOOST_ALLOW_PICKLE_TEST_CASES", "1")
    cases = [{"input": "7", "output": "8", "testtype": "stdin"}]
    encoded = base64.b64encode(zlib.compress(pickle.dumps(cases))).decode("utf-8")
    test_code_str = json.dumps([encoded])

    decoded = decode_test_cases(test_code_str)
    assert len(decoded) == 1
    assert decoded[0]["input"] == "7"


@pytest.mark.parametrize("value", ["0", "false", "False", ""])
def test_decode_test_cases_falsy_env_values_still_skip(monkeypatch, value):
    """Common falsy env values don't count as opt-in (F6)."""
    monkeypatch.setenv("SMBOOST_ALLOW_PICKLE_TEST_CASES", value)
    cases = [{"input": "1", "output": "2", "testtype": "stdin"}]
    encoded = base64.b64encode(zlib.compress(pickle.dumps(cases))).decode("utf-8")
    test_code_str = json.dumps([encoded])

    decoded = decode_test_cases(test_code_str)
    assert decoded == []

def test_sandbox_stdin_pass():
    solution = "import sys\nprint(int(sys.stdin.read().strip()) + 1)"
    test_code = json.dumps([{"input": "5\n", "output": "6\n", "testtype": "stdin"}])
    res = run(solution, test_code)
    assert res["passed"] is True
    assert res["traceback"] == ""
    assert res["duration_ms"] >= 0

def test_sandbox_stdin_fail():
    solution = "import sys\nprint(int(sys.stdin.read().strip()) + 2)"
    test_code = json.dumps([{"input": "5\n", "output": "6\n", "testtype": "stdin"}])
    res = run(solution, test_code)
    assert res["passed"] is False
    assert "Output mismatch" in res["traceback"]

def test_sandbox_functional_pass():
    solution = "class Solution:\n    def solve(self, a, b):\n        return a + b"
    entry_point = "class Solution:\n    def solve(self, a: int, b: int) -> int:\n        pass"
    test_code = json.dumps([{"input": "1\n2", "output": "3", "testtype": "functional"}])
    res = run(solution, test_code, entry_point)
    assert res["passed"] is True

def test_sandbox_timeout():
    solution = "import time\ntime.sleep(20)"
    test_code = json.dumps([{"input": "", "output": "", "testtype": "stdin"}])
    # The sandbox has a 12s timeout limit but CPU limits also apply.
    # It should timeout.
    res = run(solution, test_code)
    assert res["passed"] is False
    assert "TimeoutExpired" in res["traceback"] or "time.sleep" in res["traceback"] or "CPU" in str(res["stderr"])

def test_sandbox_syntax_error():
    solution = "def foo(:\n    pass"
    test_code = json.dumps([{"input": "", "output": "", "testtype": "stdin"}])
    res = run(solution, test_code)
    assert res["passed"] is False
    assert "SyntaxError" in res["traceback"]
