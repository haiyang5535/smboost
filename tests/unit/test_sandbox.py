import pytest
import json
import base64
import zlib
import pickle
from benchmarks.livecodebench.sandbox import run, decode_test_cases

def test_decode_test_cases():
    cases = [{"input": "1", "output": "2", "testtype": "stdin"}]
    encoded = base64.b64encode(zlib.compress(pickle.dumps(cases))).decode('utf-8')
    test_code_str = json.dumps([encoded])
    
    decoded = decode_test_cases(test_code_str)
    assert len(decoded) == 1
    assert decoded[0]["input"] == "1"

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
