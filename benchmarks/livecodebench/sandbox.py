import sys
import json
import base64
import zlib
import pickle
import subprocess
import time
import tempfile
import os

# Security: pickle.loads on untrusted data is a remote-code-execution vector.
# The LiveCodeBench-Hard dataset (downloaded from Hugging Face per
# CONTRIBUTING.md) historically encodes some test cases as pickled blobs. By
# default we refuse to deserialize them and only accept native JSON entries.
# Operators who trust the dataset source can opt back in with
# SMBOOST_ALLOW_PICKLE_TEST_CASES=1. See SECURITY.md for the trust boundary.
_PICKLE_OPT_IN_ENV = "SMBOOST_ALLOW_PICKLE_TEST_CASES"
_pickle_skip_warned = False


def _pickle_allowed() -> bool:
    return os.environ.get(_PICKLE_OPT_IN_ENV, "").strip() not in ("", "0", "false", "False")


def decode_test_cases(test_code_str: str) -> list[dict]:
    global _pickle_skip_warned
    try:
        cases = json.loads(test_code_str)
    except json.JSONDecodeError:
        return []

    decoded = []
    for c in cases:
        if isinstance(c, str):
            # Pickled payload — only decode when the operator has explicitly
            # opted in by setting SMBOOST_ALLOW_PICKLE_TEST_CASES=1.
            if not _pickle_allowed():
                if not _pickle_skip_warned:
                    print(
                        f"[smboost.sandbox] WARNING: skipping pickled LiveCodeBench test case; "
                        f"set {_PICKLE_OPT_IN_ENV}=1 to enable pickle decoding "
                        f"(see SECURITY.md for the trust boundary).",
                        file=sys.stderr,
                    )
                    _pickle_skip_warned = True
                continue
            try:
                c_dec = pickle.loads(zlib.decompress(base64.b64decode(c)))
                if isinstance(c_dec, list):
                    decoded.extend(c_dec)
                else:
                    decoded.append(c_dec)
            except Exception:
                pass
        else:
            decoded.append(c)
    return decoded

def _run_single_case(solution: str, entry_point: str, test_case: dict) -> dict:
    """Run a single test case. Returns {passed, stdout, stderr, traceback, duration_ms}"""
    testtype = test_case.get("testtype", "stdin")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "run_test.py")
        
        if testtype == "stdin":
            # For codeforces/atcoder style, just run the solution script.
            script_content = solution
            input_data = test_case.get("input", "")
            expected_output = test_case.get("output", "")
            
        elif testtype == "functional":
            # For LeetCode style.
            # `entry_point` is the class definition skeleton, e.g., "class Solution:\n    def count(self, ...):"
            # We need to extract the method name from it or assume standard behavior.
            import ast
            method_name = "solve"
            class_name = "Solution"
            try:
                parsed = ast.parse(entry_point)
                for node in parsed.body:
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                        for child in node.body:
                            if isinstance(child, ast.FunctionDef) and not child.name.startswith('_'):
                                method_name = child.name
                                break
            except Exception:
                pass
                
            input_data = test_case.get("input", "")
            expected_output = test_case.get("output", "")
            
            # The test case input string has arguments on separate lines (JSON encoded).
            # The wrapper imports json and calls the function.
            wrapper = f"""
import sys, json

{solution}

def main():
    input_str = {repr(input_data)}
    args = [json.loads(line) for line in input_str.strip().split('\\n') if line.strip()]
    
    sol = {class_name}()
    res = getattr(sol, {repr(method_name)})(*args)
    
    # Dump result as JSON to stdout for comparison
    print(json.dumps(res))

if __name__ == '__main__':
    main()
"""
            script_content = wrapper
            input_data = "" # No stdin needed
            
        else:
            return {"passed": False, "stdout": "", "stderr": f"Unknown testtype {testtype}", "traceback": "", "duration_ms": 0}
            
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
            
        start_time = time.perf_counter()
        
        # We need to use `resource` for rlimits. We can set them in a `preexec_fn`.
        def set_limits():
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
                except (ValueError, OSError):
                    pass
                try:
                    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
                except (ValueError, OSError):
                    pass
            except ImportError:
                pass # Windows
                
        try:
            # Add timeout slightly above CPU limit
            process = subprocess.run(
                [sys.executable, script_path],
                input=input_data,
                text=True,
                capture_output=True,
                timeout=12,
                preexec_fn=set_limits if sys.platform != 'win32' else None
            )
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            
            if process.returncode != 0:
                # Capture traceback from stderr
                tb = process.stderr
                return {"passed": False, "stdout": process.stdout, "stderr": process.stderr, "traceback": tb, "duration_ms": duration_ms}
                
            # Compare output
            if testtype == "stdin":
                # Strip trailing whitespace and match
                out_lines = [l.rstrip() for l in process.stdout.strip().split('\n')]
                exp_lines = [l.rstrip() for l in expected_output.strip().split('\n')]
                
                # Trim empty lines at the end
                while out_lines and not out_lines[-1]: out_lines.pop()
                while exp_lines and not exp_lines[-1]: exp_lines.pop()
                
                if out_lines == exp_lines:
                    return {"passed": True, "stdout": process.stdout, "stderr": process.stderr, "traceback": "", "duration_ms": duration_ms}
                else:
                    return {"passed": False, "stdout": process.stdout, "stderr": process.stderr, "traceback": f"Output mismatch.\nExpected:\n{expected_output}\nGot:\n{process.stdout}", "duration_ms": duration_ms}
                    
            elif testtype == "functional":
                try:
                    actual = json.loads(process.stdout.strip())
                    expected = json.loads(expected_output.strip())
                    # Some leeway for list/tuples or floats
                    if actual == expected:
                        return {"passed": True, "stdout": process.stdout, "stderr": process.stderr, "traceback": "", "duration_ms": duration_ms}
                    else:
                        return {"passed": False, "stdout": process.stdout, "stderr": process.stderr, "traceback": f"Output mismatch. Expected {expected}, got {actual}", "duration_ms": duration_ms}
                except json.JSONDecodeError:
                    return {"passed": False, "stdout": process.stdout, "stderr": process.stderr, "traceback": f"Could not parse stdout as JSON:\n{process.stdout}", "duration_ms": duration_ms}
                    
        except subprocess.TimeoutExpired as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            stdout_str = (e.stdout or b'').decode('utf-8', 'ignore') if isinstance(e.stdout, bytes) else (e.stdout or '')
            stderr_str = (e.stderr or b'').decode('utf-8', 'ignore') if isinstance(e.stderr, bytes) else (e.stderr or '')
            return {"passed": False, "stdout": stdout_str, "stderr": stderr_str, "traceback": "TimeoutExpired", "duration_ms": duration_ms}
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return {"passed": False, "stdout": "", "stderr": str(e), "traceback": str(e), "duration_ms": duration_ms}


def run(solution: str, test_code: str, entry_point: str = "") -> dict:
    """
    Executes candidate solution + test code in a fresh Python process with resource limits.
    Returns: {passed: bool, stdout: str, stderr: str, traceback: str, duration_ms: int}
    """
    test_cases = decode_test_cases(test_code)
    if not test_cases:
        return {"passed": False, "stdout": "", "stderr": "No test cases found.", "traceback": "No test cases decoded.", "duration_ms": 0}
        
    total_duration = 0
    all_stdout = ""
    all_stderr = ""
    
    for case in test_cases:
        res = _run_single_case(solution, entry_point, case)
        total_duration += res["duration_ms"]
        all_stdout += res["stdout"] + "\n"
        all_stderr += res["stderr"] + "\n"
        
        if not res["passed"]:
            return {
                "passed": False,
                "stdout": all_stdout.strip(),
                "stderr": all_stderr.strip(),
                "traceback": res["traceback"],
                "duration_ms": total_duration
            }
            
    return {
        "passed": True,
        "stdout": all_stdout.strip(),
        "stderr": all_stderr.strip(),
        "traceback": "",
        "duration_ms": total_duration
    }
