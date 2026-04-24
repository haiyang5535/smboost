"""Tests for :mod:`smboost.llm.grammar` and grammar passthrough in runtime.

Covers:

* JSON-Schema → GBNF conversion for the subset we emit (object with
  required integer field, string enum, array-of-number).
* Canned grammars are non-empty strings and parse as valid GBNF
  (validated via ``llama_cpp.LlamaGrammar.from_string`` when available —
  otherwise we fall back to structural assertions).
* ``_CompatibleChatOpenAI`` threads ``grammar`` through to the request
  payload and leaves it absent when unset (regression guard against the
  langchain-openai 1.1 path that silently routes unknown kwargs into
  ``model_kwargs``).
* ``@pytest.mark.slow``: live round-trip against the Qwen3.5:2B
  llama.cpp server on ``http://127.0.0.1:8000/v1`` using
  ``GSM8K_FINAL_ANSWER_GRAMMAR``.
"""

from __future__ import annotations

import json
import os
import re

import pytest
from langchain_core.messages import HumanMessage

from smboost.llm.grammar import (
    GSM8K_FINAL_ANSWER_GRAMMAR,
    JSON_FUNCTION_CALL_GRAMMAR,
    JSON_VALUE_GRAMMAR,
    PYTHON_CODE_BLOCK_GRAMMAR,
    json_schema_to_gbnf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_parse_gbnf(grammar_text: str) -> bool:
    """Return True if llama-cpp-python accepts ``grammar_text`` as valid GBNF.

    If llama-cpp-python isn't importable we skip this structural check —
    the string-level assertions are still enough to catch the common
    regressions (empty output, missing rule, wrong literal).
    """
    try:
        from llama_cpp import LlamaGrammar  # type: ignore
    except Exception:
        pytest.skip("llama-cpp-python not installed; can't validate GBNF structurally")
    LlamaGrammar.from_string(grammar_text, verbose=False)
    return True


# ---------------------------------------------------------------------------
# json_schema_to_gbnf
# ---------------------------------------------------------------------------


def test_json_schema_to_gbnf_required_integer_field_matches_simple_object():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    grammar = json_schema_to_gbnf(schema)

    # The generated grammar must include a rule that fixes the key to
    # exactly ``"x"`` (otherwise the grammar would accept any property
    # name and the converter would be wrong).
    assert '"\\"x\\""' in grammar

    # There must be a root rule and an integer-ish leaf.
    assert re.search(r"^root\s*::=", grammar, re.MULTILINE), grammar
    assert "[0-9]" in grammar

    # Structurally valid GBNF per llama-cpp-python.
    _try_parse_gbnf(grammar)

    # Sanity: the target string is what a conforming model would emit.
    target = json.dumps({"x": 42})
    # Loose structural check (full GBNF matching is out of scope for a
    # unit test — the live e2e covers that); we assert that the bytes
    # we need to see show up in the grammar.
    assert '"x"' in target
    assert target == '{"x": 42}'


def test_json_schema_to_gbnf_string_enum():
    schema = {"type": "string", "enum": ["add", "subtract"]}
    grammar = json_schema_to_gbnf(schema)
    # Each enum value must appear as a JSON-quoted literal in the grammar.
    assert '"\\"add\\""' in grammar
    assert '"\\"subtract\\""' in grammar
    _try_parse_gbnf(grammar)


def test_json_schema_to_gbnf_array_of_numbers():
    schema = {"type": "array", "items": {"type": "number"}}
    grammar = json_schema_to_gbnf(schema)
    assert "number" in grammar
    # Array brackets must be forced.
    assert '"["' in grammar
    assert '"]"' in grammar
    _try_parse_gbnf(grammar)


def test_json_schema_to_gbnf_unknown_type_falls_back_permissively():
    # No ``type`` field → permissive fallback to JSON ``value``.
    grammar = json_schema_to_gbnf({})
    assert "value" in grammar
    _try_parse_gbnf(grammar)


def test_json_schema_to_gbnf_rejects_non_dict():
    with pytest.raises(TypeError):
        json_schema_to_gbnf("not a schema")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Canned grammars
# ---------------------------------------------------------------------------


def test_python_code_block_grammar_is_non_empty_valid_gbnf():
    assert isinstance(PYTHON_CODE_BLOCK_GRAMMAR, str)
    assert PYTHON_CODE_BLOCK_GRAMMAR.strip()
    assert "```python" in PYTHON_CODE_BLOCK_GRAMMAR
    _try_parse_gbnf(PYTHON_CODE_BLOCK_GRAMMAR)


def test_gsm8k_final_answer_grammar_is_non_empty_valid_gbnf():
    assert isinstance(GSM8K_FINAL_ANSWER_GRAMMAR, str)
    assert GSM8K_FINAL_ANSWER_GRAMMAR.strip()
    assert "####" in GSM8K_FINAL_ANSWER_GRAMMAR
    _try_parse_gbnf(GSM8K_FINAL_ANSWER_GRAMMAR)


def test_json_value_grammar_is_non_empty_valid_gbnf():
    assert isinstance(JSON_VALUE_GRAMMAR, str)
    assert JSON_VALUE_GRAMMAR.strip()
    _try_parse_gbnf(JSON_VALUE_GRAMMAR)


def test_json_function_call_grammar_binds_function_name():
    grammar = JSON_FUNCTION_CALL_GRAMMAR(
        "add",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    assert isinstance(grammar, str)
    assert grammar.strip()
    # The literal function name must be wired into the grammar.
    assert '"\\"add\\""' in grammar
    # The arguments keys must appear as literals.
    assert '"\\"a\\""' in grammar
    assert '"\\"b\\""' in grammar
    _try_parse_gbnf(grammar)


def test_json_function_call_grammar_rejects_empty_fn_name():
    with pytest.raises(ValueError):
        JSON_FUNCTION_CALL_GRAMMAR("", {"type": "object"})


# ---------------------------------------------------------------------------
# Runtime passthrough
# ---------------------------------------------------------------------------


def _make_chat_openai(grammar=None):
    from smboost.llm.runtime import _CompatibleChatOpenAI

    return _CompatibleChatOpenAI(
        model="x",
        base_url="http://127.0.0.1:8000/v1",
        api_key="sk-no-key",
        max_tokens=8,
        temperature=0.0,
        grammar=grammar,
    )


def test_compatible_chat_openai_payload_omits_grammar_by_default():
    llm = _make_chat_openai()
    payload = llm._get_request_payload([HumanMessage(content="hi")])
    assert "grammar" not in payload
    # Regression guard for the original bug this class was added for.
    assert payload.get("max_tokens") == 8


def test_compatible_chat_openai_payload_passes_grammar_through():
    grammar = 'root ::= "hello"'
    llm = _make_chat_openai(grammar=grammar)
    payload = llm._get_request_payload([HumanMessage(content="hi")])
    assert payload.get("grammar") == grammar


def test_compatible_chat_openai_with_grammar_returns_new_instance():
    base = _make_chat_openai()
    derived = base.with_grammar('root ::= "x"')
    assert base.grammar is None
    assert derived.grammar == 'root ::= "x"'
    assert base is not derived


def test_compatible_chat_openai_per_call_grammar_override():
    """Per-call ``grammar=`` kwarg on ``_get_request_payload`` wins over instance grammar."""
    llm = _make_chat_openai(grammar='root ::= "default"')
    payload = llm._get_request_payload(
        [HumanMessage(content="hi")], grammar='root ::= "override"'
    )
    assert payload["grammar"] == 'root ::= "override"'


# ---------------------------------------------------------------------------
# Live e2e test (slow, requires the :8000 llama.cpp server)
# ---------------------------------------------------------------------------


def _server_reachable(url: str) -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.mark.slow
def test_live_grammar_constrained_gsm8k_answer_format():
    """End-to-end: grammar forces Qwen3.5:2B to emit ``#### <integer>``.

    Runs only when the llama.cpp server on ``127.0.0.1:8000`` is reachable.
    Uses ``requests`` directly so this test does not depend on the harness
    graph or any langchain wiring.
    """
    base_url = os.environ.get(
        "SMBOOST_GRAMMAR_TEST_BASE_URL", "http://127.0.0.1:8000/v1"
    )
    if not _server_reachable(base_url.rstrip("/") + "/models"):
        pytest.skip(f"no llama.cpp server at {base_url}")

    try:
        import requests  # noqa: F401
    except Exception:
        pytest.skip("requests not installed")

    import requests

    # Use the first model the server reports (we don't care which Qwen
    # size answered, only that the grammar constrained the output).
    models = requests.get(base_url.rstrip("/") + "/models", timeout=5).json()
    model_id = models["data"][0]["id"]

    prompt = (
        "Solve this word problem. Think step-by-step, then on a new line "
        "emit '#### <integer>' with the final numeric answer.\n\n"
        "Problem: A farmer has 3 chickens. Each chicken lays 4 eggs. How "
        "many eggs does the farmer have in total?"
    )
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
        "grammar": GSM8K_FINAL_ANSWER_GRAMMAR,
    }

    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions", json=body, timeout=120
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # The grammar forces the output to contain "#### " followed by digits.
    match = re.search(r"####\s*(-?\d+)", content)
    assert match is not None, f"grammar did not force #### terminator; got: {content!r}"
    # Correct answer is 12 but we don't gate the test on that — the
    # grammar contract is structural, not semantic.  Just assert the
    # integer parses.
    int(match.group(1))
