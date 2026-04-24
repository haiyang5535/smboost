from __future__ import annotations
import json
from typing import Callable

from smboost.harness.state import HarnessState

InvariantFn = Callable[[HarnessState, str | None], bool]

_ERROR_MARKERS = ("Error:", "Traceback", "Exception:")


def output_is_nonempty(state: HarnessState, output: str | None) -> bool:
    if output is None:
        return True
    return len(output.strip()) > 0


def output_is_valid_json(state: HarnessState, output: str | None) -> bool:
    if output is None:
        return True
    try:
        json.loads(output)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def no_error_keywords(state: HarnessState, output: str | None) -> bool:
    if output is None:
        return True
    return not any(marker in output for marker in _ERROR_MARKERS)


def verify_says_pass(state: "HarnessState", output: str | None) -> bool:
    return output is not None and output.upper().startswith("PASS")


class InvariantSuite:
    def __init__(
        self,
        node_invariants: dict[str, tuple[list[InvariantFn], list[InvariantFn]]],
    ):
        self.node_invariants = node_invariants

    @staticmethod
    def coding_agent() -> InvariantSuite:
        return InvariantSuite({
            "plan":    ([], [output_is_nonempty]),
            "execute": ([], [output_is_nonempty, no_error_keywords]),
            "verify":  ([], [output_is_nonempty, verify_says_pass]),
        })

    @staticmethod
    def tool_calling() -> InvariantSuite:
        return InvariantSuite({
            "plan":     ([], [output_is_nonempty]),
            "dispatch": ([], [output_is_nonempty]),
            "verify":   ([], [output_is_nonempty, verify_says_pass]),
        })

    @staticmethod
    def completion() -> InvariantSuite:
        return InvariantSuite({
            "generate": ([], [output_is_nonempty]),
            "verify":   ([], [output_is_nonempty, verify_says_pass]),
        })

    @staticmethod
    def real_tool() -> InvariantSuite:
        """Invariants for the C6 real-tool graph (plan → call → verify).

        plan and call must emit non-empty strings (they are JSON blobs).
        verify is PASS/FAIL.
        """
        return InvariantSuite({
            "plan":   ([], [output_is_nonempty]),
            "call":   ([], [output_is_nonempty]),
            "verify": ([], [output_is_nonempty, verify_says_pass]),
        })

    @staticmethod
    def emit_only_tool_calling() -> InvariantSuite:
        """Invariants for the emit-only tool-calling graph (BFCL).

        The generate node must produce nonempty output (a JSON call). The verify
        node is structural-only (PASS/FAIL), so we demand it says PASS — but we
        do NOT require the generate output to be valid JSON as an invariant,
        because the verify node itself produces the authoritative structural
        judgement and a soft-fail there should trigger retry/shrinkage rather
        than an invariant violation on `generate`.
        """
        return InvariantSuite({
            "plan":     ([], [output_is_nonempty]),
            "generate": ([], [output_is_nonempty]),
            "verify":   ([], [output_is_nonempty, verify_says_pass]),
        })
