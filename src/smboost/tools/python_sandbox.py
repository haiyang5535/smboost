"""Subprocess-isolated Python execution sandbox.

Used by the C6 real-tool harness (and, in principle, any other place that needs
to actually *execute* a piece of model-authored code without letting it touch
the harness process).

Design notes
------------
- Each call spawns a fresh `python3 -c "..."` subprocess. Timeout is enforced
  via ``subprocess.run(timeout=...)``.
- stdout and stderr are captured. If code `print(expr)`s we record it; if code
  raises we record the traceback text on stderr.
- A ``result`` channel is supported: if the submitted code assigns to a
  top-level variable named ``result``, the sandbox appends a small trailer
  that prints ``repr(result)`` to stdout (wrapped with a sentinel) so callers
  can recover the in-band return value. This is best-effort; if the code does
  not define ``result`` the ``result`` key in the output will be ``None``.
- Working directory is a fresh ``tempfile.mkdtemp`` that is removed after the
  call (best-effort — a rogue subprocess cannot prevent cleanup because we
  kill it on timeout).
- On non-macOS POSIX we attempt ``resource.setrlimit(RLIMIT_AS, ...)`` via a
  Python preamble so the child can't allocate gigabytes of RAM. This is
  *gracefully skipped* if ``setrlimit`` raises (py3.13 on mac has a known
  flaky behaviour around RLIMIT_AS — see CLAUDE.md note about evalplus).

Return shape::

    {
      "ok":     bool,          # False iff subprocess errored, timed out, or returncode != 0
      "stdout": str,           # with sentinel-wrapped `result` trailer stripped
      "stderr": str,
      "result": str | None,    # repr() of a top-level `result` variable, if any
      "error":  str | None,    # short human-readable error tag: "timeout", "nonzero_exit", ...
    }
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

_RESULT_START = "__SMBOOST_RESULT_START__"
_RESULT_END = "__SMBOOST_RESULT_END__"

# A preamble that:
#   1. best-effort tightens the address-space limit
#   2. installs a sys.excepthook that writes the traceback to stderr and exits 1
#   3. runs the user code inside an exec() against a fresh dict
#   4. if the user code assigned to `result`, prints a sentinel-wrapped repr()
_PREAMBLE = r"""
import sys, traceback

try:
    import resource
    # 512 MB address space; skip silently if the platform rejects it.
    try:
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    except (ValueError, OSError):
        pass
except ImportError:
    pass

_SMBOOST_NS = {{"__name__": "__main__"}}
_SMBOOST_SRC = {user_code!r}

try:
    exec(compile(_SMBOOST_SRC, "<sandbox>", "exec"), _SMBOOST_NS)
except SystemExit:
    raise
except BaseException:
    traceback.print_exc()
    sys.exit(1)

if "result" in _SMBOOST_NS:
    sys.stdout.write({start!r})
    try:
        sys.stdout.write(repr(_SMBOOST_NS["result"]))
    except BaseException as _exc:
        sys.stdout.write("<unreprable: " + type(_exc).__name__ + ">")
    sys.stdout.write({end!r})
    sys.stdout.write("\n")
"""


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    result: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "result": self.result,
            "error": self.error,
        }


def _split_result(stdout: str) -> tuple[str, str | None]:
    """Strip the sentinel-wrapped `result` trailer from stdout, returning
    (clean_stdout, result_repr_or_None)."""
    start_idx = stdout.rfind(_RESULT_START)
    if start_idx < 0:
        return stdout, None
    end_idx = stdout.find(_RESULT_END, start_idx)
    if end_idx < 0:
        return stdout, None
    result_repr = stdout[start_idx + len(_RESULT_START) : end_idx]
    # Drop trailing newline written after the end sentinel, if present.
    tail = stdout[end_idx + len(_RESULT_END) :]
    if tail.startswith("\n"):
        tail = tail[1:]
    clean = stdout[:start_idx] + tail
    return clean, result_repr


class PythonSandbox:
    """Subprocess-based Python code executor.

    Usage::

        sb = PythonSandbox()
        out = sb.run("print(2+2)")
        assert out["ok"] and out["stdout"].strip() == "4"

    The instance is stateless; a single ``PythonSandbox()`` can be reused for
    many calls. Each ``run()`` spawns a fresh subprocess; no state persists
    between calls.
    """

    def __init__(self, python_executable: str | None = None) -> None:
        self._python = python_executable or sys.executable or "python3"

    def run(self, code: str, timeout_s: float = 5.0) -> dict[str, Any]:
        """Execute ``code`` in a fresh subprocess.

        Args:
            code:       Python source to execute. Runs at module level; assign
                        to a top-level ``result`` variable to capture a return
                        value in the result channel.
            timeout_s:  Wall-clock timeout. On expiry the subprocess is killed
                        and ``run()`` returns ``ok=False, error="timeout"``.

        Returns:
            A dict matching ``SandboxResult.to_dict()``.
        """
        workdir = tempfile.mkdtemp(prefix="smboost_sbx_")
        try:
            preamble = _PREAMBLE.format(
                user_code=code,
                start=_RESULT_START,
                end=_RESULT_END,
            )
            try:
                completed = subprocess.run(
                    # -I = isolated (no PYTHON* env, no user site, no cwd in path)
                    # -S = skip `import site` (no system site-packages on path)
                    # Combined, they guarantee the tool subprocess cannot reach
                    # the harness's editable-install smboost package.
                    [self._python, "-I", "-S", "-c", preamble],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=workdir,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                clean_stdout, _ = _split_result(stdout)
                return SandboxResult(
                    ok=False,
                    stdout=clean_stdout,
                    stderr=stderr,
                    result=None,
                    error="timeout",
                ).to_dict()
            except FileNotFoundError as exc:
                return SandboxResult(
                    ok=False,
                    stdout="",
                    stderr=str(exc),
                    result=None,
                    error="interpreter_not_found",
                ).to_dict()

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            clean_stdout, result_repr = _split_result(stdout)

            if completed.returncode == 0:
                return SandboxResult(
                    ok=True,
                    stdout=clean_stdout,
                    stderr=stderr,
                    result=result_repr,
                    error=None,
                ).to_dict()

            return SandboxResult(
                ok=False,
                stdout=clean_stdout,
                stderr=stderr,
                result=result_repr,
                error=f"nonzero_exit:{completed.returncode}",
            ).to_dict()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


__all__ = ["PythonSandbox", "SandboxResult"]
