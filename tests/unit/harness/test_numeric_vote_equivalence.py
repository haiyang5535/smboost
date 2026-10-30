import pytest

from benchmarks.gsm8k.scorer import extract_answer, score
from smboost.harness.self_consistency_graph import execute_program_verifier, majority_vote


@pytest.mark.parametrize("token, expected", [
    ("1,000", 1000),
    ("-1,200", -1200),
    ("1,234,567", 1234567),
    ("9007199254740993", 9007199254740993),
    ("-9007199254740993", -9007199254740993),
    ("999,999,999,999,999,999", 999999999999999999),
])
@pytest.mark.parametrize("output_kind", ["marker", "program"])
def test_numeric_vote_key_preserves_integer_value(token, expected, output_kind):
    completion = f"#### {token}" if output_kind == "marker" else f"print({token!r})"
    result = execute_program_verifier(completion, {})
    assert result["valid"]
    assert result["extracted_answer"] == expected
    assert int(extract_answer(f"#### {token}")) == expected


@pytest.mark.parametrize("token, expected", [("1,200.5", 1200.5), ("-1,200.5", -1200.5)])
def test_grouped_decimal_output(token, expected):
    result = execute_program_verifier(f"print({token!r})", {})
    assert result["valid"]
    assert result["extracted_answer"] == expected


def test_formatting_does_not_reverse_majority():
    samples = ["#### 1", "#### 1", "#### 1000", "#### 1,000", "#### 1,000"]
    results = [execute_program_verifier(sample, {}) for sample in samples]
    chosen, answer, valid, votes = majority_vote(samples, results)
    assert (answer, valid, votes) == (1000, 5, 3)
    assert score(chosen, "1000")


def test_adjacent_large_integers_are_distinct_votes():
    samples = ["#### 9007199254740992", "#### 9007199254740993", "#### 9007199254740993"]
    results = [execute_program_verifier(sample, {}) for sample in samples]
    chosen, answer, valid, votes = majority_vote(samples, results)
    assert (answer, valid, votes) == (9007199254740993, 3, 2)
    assert score(chosen, "9007199254740993")
