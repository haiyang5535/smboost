from smboost.harness.state import HarnessState
from smboost.invariants.suite import (
    InvariantSuite,
    output_is_nonempty,
    output_is_valid_json,
    no_error_keywords,
    verify_says_pass,
)

def _state() -> HarnessState:
    return {
        "task": "t", "model": "qwen3.5:2b", "fallback_chain": [],
        "step_outputs": [], "retry_count": 0, "fallback_index": 0,
        "current_node_index": 0, "status": "running", "final_output": None,
    }

# output_is_nonempty
def test_nonempty_passes_for_content():
    assert output_is_nonempty(_state(), "hello") is True

def test_nonempty_fails_for_whitespace_only():
    assert output_is_nonempty(_state(), "   ") is False

def test_nonempty_passes_for_none_entry_invariant():
    assert output_is_nonempty(_state(), None) is True

# output_is_valid_json
def test_valid_json_passes():
    assert output_is_valid_json(_state(), '{"key": "value"}') is True

def test_valid_json_fails_for_plain_text():
    assert output_is_valid_json(_state(), "not json") is False

def test_valid_json_passes_for_none():
    assert output_is_valid_json(_state(), None) is True

# no_error_keywords
def test_no_error_keywords_passes_for_clean_output():
    assert no_error_keywords(_state(), "task completed successfully") is True

def test_no_error_keywords_fails_for_traceback():
    assert no_error_keywords(_state(), "Traceback (most recent call last):\n  File...") is False

def test_no_error_keywords_fails_for_error_prefix():
    assert no_error_keywords(_state(), "Error: file not found") is False

def test_no_error_keywords_passes_for_none():
    assert no_error_keywords(_state(), None) is True

# InvariantSuite factory methods
def test_coding_agent_suite_has_expected_nodes():
    suite = InvariantSuite.coding_agent()
    assert "plan" in suite.node_invariants
    assert "execute" in suite.node_invariants
    assert "verify" in suite.node_invariants

def test_tool_calling_suite_has_expected_nodes():
    suite = InvariantSuite.tool_calling()
    assert "plan" in suite.node_invariants
    assert "dispatch" in suite.node_invariants
    assert "verify" in suite.node_invariants

def test_invariant_fn_type_alias_is_importable():
    from smboost.invariants.suite import InvariantFn
    assert InvariantFn is not None

# verify_says_pass
def test_verify_says_pass_returns_true_for_pass_prefix():
    assert verify_says_pass(_state(), "PASS: looks good") is True

def test_verify_says_pass_returns_false_for_fail_prefix():
    assert verify_says_pass(_state(), "FAIL: something broke") is False

def test_verify_says_pass_returns_false_for_none():
    assert verify_says_pass(_state(), None) is False
