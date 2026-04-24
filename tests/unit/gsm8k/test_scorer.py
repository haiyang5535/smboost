from __future__ import annotations

from benchmarks.gsm8k.scorer import extract_answer, score


# ----- extract_answer --------------------------------------------------------


def test_extract_hash_marker_simple():
    assert extract_answer("step by step... #### 42") == "42"


def test_extract_hash_marker_negative():
    assert extract_answer("... #### -7") == "-7"


def test_extract_hash_marker_comma_grouped():
    assert extract_answer("result is #### 1,200") == "1200"


def test_extract_prefers_last_marker_few_shot_contamination():
    """Few-shot leakage: model sometimes echoes an example '#### 5' before its
    own final '#### 9'. Take the last."""
    text = "Example: #### 5. Now the real answer is: #### 9"
    assert extract_answer(text) == "9"


def test_extract_rejects_decimal_marker():
    """#### 4.5 → None (GSM8K answers are integers; decimal means wrong)."""
    assert extract_answer("reasoning... #### 4.5") is None


def test_extract_falls_back_to_last_integer_when_marker_missing():
    assert extract_answer("The answer is 42. Hope this helps!") == "42"


def test_extract_fallback_uses_last_integer_not_first():
    """'got 5, added 3, final 8' → 8."""
    assert extract_answer("got 5, added 3, final 8") == "8"


def test_extract_falls_back_when_marker_is_decimal():
    """If the '#### <N>' marker matches but the number is a decimal (hence
    rejected), the fallback should kick in and pick up a prior integer token."""
    assert extract_answer("The answer should be 42 but I'll write #### 4.2") == "42"


def test_extract_returns_none_for_empty_string():
    assert extract_answer("") is None


def test_extract_returns_none_for_pure_prose():
    assert extract_answer("I don't know the answer sorry") is None


def test_extract_zero():
    assert extract_answer("... #### 0") == "0"


def test_extract_negative_zero_normalizes_to_zero():
    assert extract_answer("... #### -0") == "0"


def test_extract_strips_leading_zeros():
    assert extract_answer("... #### 007") == "7"


# ----- score -----------------------------------------------------------------


def test_score_positive_match():
    assert score("blah #### 42", "42") is True


def test_score_negative_mismatch():
    assert score("blah #### 41", "42") is False


def test_score_comma_normalization_both_sides():
    assert score("#### 1200", "1,200") is True


def test_score_returns_false_when_no_answer():
    assert score("prose only", "42") is False


def test_score_tolerates_extra_whitespace_in_expected():
    assert score("#### 7", "  7  ") is True


def test_score_negative_number_match():
    assert score("so the answer is #### -3", "-3") is True


def test_score_decimal_completion_does_not_match_integer_expected():
    """If the model emits a decimal, that's treated as non-answer and fails."""
    assert score("#### 3.0", "3") is False


def test_score_returns_false_for_empty_expected():
    assert score("#### 5", "") is False
