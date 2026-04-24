"""GSM8K scorer — numeric final-answer extraction + normalized compare.

The scorer is deliberately simple (numeric equality) because GSM8K's
official evaluation is exact-match on the integer final answer. A future
verifier hook (Prove-style program verifier) can wrap ``score`` without
changing its signature.

Extraction order:
    1. Last ``#### <N>`` marker in the completion (canonical)
    2. Fallback: last integer token in the text (tolerates small models
       that forgot the marker but still produced a number)

Edge-case policy:
    - Commas stripped ("1,200" → "1200")
    - Negative integers accepted ("-7")
    - Decimals REJECTED — GSM8K answers are always integers; a decimal in
      the output signals the model computed something else (rounding,
      percentages treated as fractions, etc.)
"""
from __future__ import annotations

import re


# Canonical marker: "#### <number>" — prefer the *last* match in case the
# completion includes a few-shot example or repeats the marker inside
# reasoning.
_MARKER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
# Generic numeric token, capturing the full comma/decimal form so the
# normalizer can reject decimals (matching "2" inside "4.2" would otherwise
# pass scoring). Matching order below walks this tail-first.
_NUM_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize(token: str | None) -> str | None:
    """Strip commas/whitespace; reject decimals. None-in → None-out."""
    if token is None:
        return None
    t = token.strip().replace(",", "")
    if not t:
        return None
    # GSM8K answers are integers; a "." signals model computed wrong thing.
    if "." in t:
        return None
    # Validate shape: optional leading '-' then digits.
    if not re.fullmatch(r"-?\d+", t):
        return None
    # Canonicalize (strip leading zeros but keep sign; "007" → "7", "-0" → "0").
    sign = "-" if t.startswith("-") else ""
    digits = t.lstrip("-").lstrip("0") or "0"
    if digits == "0":
        return "0"
    return sign + digits


def extract_answer(completion: str) -> str | None:
    """Return the normalized numeric answer extracted from ``completion``.

    Returns ``None`` if no usable integer can be found (e.g. model emitted
    only prose, emitted a decimal, or the text is empty).
    """
    if not completion:
        return None

    # 1. Prefer the LAST "#### <N>" marker (handles few-shot contamination).
    markers = _MARKER_RE.findall(completion)
    if markers:
        norm = _normalize(markers[-1])
        if norm is not None:
            return norm
        # Marker present but decimal/invalid — fall through to integer fallback.

    # 2. Fallback: last numeric token in the text. Decimals are captured
    #    in full so they get rejected by _normalize (rather than silently
    #    matching the trailing digits, e.g. "4.2" -> "2").
    nums = _NUM_TOKEN_RE.findall(completion)
    if nums:
        for token in reversed(nums):
            norm = _normalize(token)
            if norm is not None:
                return norm

    return None


def score(completion: str, expected: str) -> bool:
    """True iff ``completion``'s extracted answer normalizes-equal to ``expected``."""
    got = extract_answer(completion)
    want = _normalize(expected)
    if got is None or want is None:
        return False
    return got == want
