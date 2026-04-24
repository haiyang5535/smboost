"""GBNF grammar support for guided decoding with llama.cpp.

Small open-weight models (Qwen3.5 0.8B/2B, Phi-4-Mini) frequently produce
outputs that are *almost* in the right format but fail a strict parser
(missing code fence, extra prose after JSON, wrong casing on ``#### 42``).
llama.cpp's server supports a ``grammar`` request-body field that
constrains every sampled token to satisfy a GBNF grammar.  Published
results show 10-100x reduction in structural/format errors for small
models vs unconstrained sampling.

This module ships:
  * :func:`json_schema_to_gbnf` - a best-effort JSON-Schema → GBNF
    converter covering the subset we actually emit for tool-call payloads
    (object / integer / number / string / boolean / enum / array).
  * Canned grammars for the three output shapes the harness uses:
    Python code blocks, JSON function calls, GSM8K final answer.

The converter intentionally stays dependency-free.  For anything more
exotic (``$ref``, ``oneOf`` with dispatching, patternProperties) we fall
back to a permissive ``json`` nonterminal rather than silently producing
a wrong grammar.

GBNF reference:
    https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
llama-cpp-python server docs:
    https://github.com/abetlen/llama-cpp-python
"""

from __future__ import annotations

import json as _json
from typing import Any

__all__ = [
    "PYTHON_CODE_BLOCK_GRAMMAR",
    "GSM8K_FINAL_ANSWER_GRAMMAR",
    "JSON_VALUE_GRAMMAR",
    "JSON_FUNCTION_CALL_GRAMMAR",
    "json_schema_to_gbnf",
]


# ---------------------------------------------------------------------------
# Core JSON grammar (reused by every JSON-shaped canned grammar below).
#
# This is the "reference" JSON grammar from the GBNF README, lightly
# reformatted.  We keep it as a module-level constant rather than inlining
# it per-grammar so callers can compose additional rules on top of a
# single value vocabulary.
# ---------------------------------------------------------------------------

_JSON_CORE_RULES = r"""
value  ::= object | array | string | number | boolean | null
object ::= "{" ws ( string ws ":" ws value ( ws "," ws string ws ":" ws value )* )? ws "}"
array  ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string ::= "\"" ( [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
boolean ::= "true" | "false"
null    ::= "null"
ws      ::= ([ \t\n])*
""".strip()


JSON_VALUE_GRAMMAR = "root ::= value\n" + _JSON_CORE_RULES
"""Accept any valid JSON document at the root."""


# ---------------------------------------------------------------------------
# Canned grammars
# ---------------------------------------------------------------------------

# Forces the model to emit exactly ```python\n<code>\n```.
#
# We accept any non-backtick bytes inside the fence so normal Python code
# passes.  Triple-backticks cannot appear inside (which is what we want —
# the LLM sometimes emits nested fences and breaks downstream parsers).
PYTHON_CODE_BLOCK_GRAMMAR = r"""
root   ::= "```python\n" body "\n```"
body   ::= char*
char   ::= [^`] | "`" [^`] | "``" [^`]
""".strip()


# Forces the model to emit exactly "#### <integer>" *somewhere*, preceded
# by arbitrary reasoning prose.  This matches the standard GSM8K answer
# format ("… so the final answer is 42.\n#### 42") — we only constrain
# the terminator.
GSM8K_FINAL_ANSWER_GRAMMAR = r"""
root    ::= prose "#### " integer ws
prose   ::= char*
char    ::= [^#] | "#" [^#] | "##" [^#] | "###" [^#]
integer ::= "-"? [0-9]+
ws      ::= [ \t\n]*
""".strip()


def JSON_FUNCTION_CALL_GRAMMAR(fn_name: str, params_schema: dict[str, Any]) -> str:
    """Build a grammar that forces ``{"name": "<fn_name>", "arguments": <schema>}``.

    ``params_schema`` is the JSON Schema for the ``arguments`` object;
    we route it through :func:`json_schema_to_gbnf` so the enclosed
    arguments object must itself conform.  Everything else is fixed
    string literals (name key, fn_name literal, arguments key,
    surrounding braces) — the model cannot hallucinate a different
    function name.

    The returned grammar is self-contained (includes JSON core rules
    under the ``arg_`` namespace).
    """
    if not fn_name or not isinstance(fn_name, str):
        raise ValueError("fn_name must be a non-empty string")
    # Quote the function name as a JSON string literal so any embedded
    # quotes or backslashes are escaped correctly.
    fn_name_literal = _json.dumps(fn_name)

    # Build the arguments-object grammar in its own namespace.  We prefix
    # every rule it generates with ``arg_`` so it can't collide with our
    # root/header.
    arg_grammar = _schema_to_gbnf_rules(params_schema, prefix="arg_", root_name="arg_root")

    header = (
        'root ::= "{" ws "\\"name\\"" ws ":" ws '
        f'"{_escape_for_gbnf_literal(fn_name_literal)}"'
        ' ws "," ws "\\"arguments\\"" ws ":" ws arg_root ws "}"'
    )
    return header + "\n" + arg_grammar


# ---------------------------------------------------------------------------
# JSON Schema → GBNF conversion
# ---------------------------------------------------------------------------


def json_schema_to_gbnf(schema: dict[str, Any]) -> str:
    """Convert a JSON Schema dict to a GBNF grammar string.

    Supported subset:

    * ``{"type": "object"}`` with ``properties`` + ``required``
    * ``{"type": "array"}`` with ``items``
    * ``{"type": "string"}`` (optionally with ``enum``)
    * ``{"type": "integer" | "number" | "boolean" | "null"}``
    * ``{"enum": [...]}`` for any scalar enum

    Anything else falls back to the permissive ``value`` nonterminal
    (any JSON value).  That is deliberately conservative — a too-loose
    grammar is still *correct*; a wrong grammar would silently reject
    valid tool calls.
    """
    if not isinstance(schema, dict):
        raise TypeError("schema must be a dict (JSON Schema)")
    rules = _schema_to_gbnf_rules(schema, prefix="r_", root_name="root")
    return rules


def _schema_to_gbnf_rules(
    schema: dict[str, Any], *, prefix: str, root_name: str
) -> str:
    """Build a complete grammar whose entry point is ``root_name``."""
    builder = _GbnfBuilder(prefix=prefix)
    builder.add_rule(root_name, builder.schema_rhs(schema))
    # Always include the JSON core rules (value/object/array/string/
    # number/boolean/null/ws) so that fallback branches resolve.
    out = builder.render()
    return out + "\n" + _JSON_CORE_RULES


class _GbnfBuilder:
    def __init__(self, *, prefix: str) -> None:
        self._prefix = prefix
        self._rules: list[tuple[str, str]] = []
        self._counter = 0

    def fresh(self, hint: str) -> str:
        # Sanitize hint to [a-z0-9_].
        safe = "".join(c if c.isalnum() else "_" for c in hint.lower())
        self._counter += 1
        return f"{self._prefix}{safe}_{self._counter}"

    def add_rule(self, name: str, rhs: str) -> None:
        self._rules.append((name, rhs))

    def render(self) -> str:
        return "\n".join(f"{name} ::= {rhs}" for name, rhs in self._rules)

    # ------------------------------------------------------------------
    # Schema → RHS
    # ------------------------------------------------------------------

    def schema_rhs(self, schema: dict[str, Any]) -> str:
        # enum short-circuits everything else.
        if "enum" in schema:
            alternatives = [_json_literal_rhs(v) for v in schema["enum"]]
            return " | ".join(alternatives) if alternatives else "value"

        t = schema.get("type")
        if t == "object":
            return self._object_rhs(schema)
        if t == "array":
            return self._array_rhs(schema)
        if t == "string":
            return "string"
        if t == "integer":
            # GBNF integers: optional sign, digits.
            return '"-"? ([0-9] | [1-9] [0-9]+)'
        if t == "number":
            return "number"
        if t == "boolean":
            return "boolean"
        if t == "null":
            return "null"
        # Unknown / unconstrained → any JSON value.
        return "value"

    def _object_rhs(self, schema: dict[str, Any]) -> str:
        props = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        if not props:
            return "object"

        # We emit required properties in the order they appear in ``required``
        # (or, if that list is missing, in property-declaration order).
        # Optional properties are not currently emitted — a too-tight grammar
        # for optionals is worse than omitting them, because the model will
        # then just fail to produce optional fields ever.  For our first
        # use case (tool-call arguments), every field we care about should
        # be required.
        if required:
            order = required
        else:
            order = list(props.keys())

        parts: list[str] = ['"{"', "ws"]
        for i, key in enumerate(order):
            if i > 0:
                parts.extend(['ws', '","', "ws"])
            key_literal = _escape_for_gbnf_literal(_json.dumps(key))
            value_rule = self._subschema_nonterminal(props.get(key, {}), hint=key)
            parts.extend([f'"{key_literal}"', "ws", '":"', "ws", value_rule])
        parts.extend(["ws", '"}"'])
        return " ".join(parts)

    def _array_rhs(self, schema: dict[str, Any]) -> str:
        items = schema.get("items")
        if not isinstance(items, dict):
            return "array"
        item_rule = self._subschema_nonterminal(items, hint="item")
        return (
            '"[" ws ( ' + item_rule + ' ( ws "," ws ' + item_rule + " )* )? ws \"]\""
        )

    def _subschema_nonterminal(self, subschema: dict[str, Any], *, hint: str) -> str:
        """Emit a fresh nonterminal for ``subschema`` and return its name."""
        rhs = self.schema_rhs(subschema)
        name = self.fresh(hint)
        self.add_rule(name, rhs)
        return name


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _json_literal_rhs(value: Any) -> str:
    """Return a GBNF RHS that matches exactly the JSON encoding of ``value``.

    Used for enum alternatives.
    """
    encoded = _json.dumps(value)
    return f'"{_escape_for_gbnf_literal(encoded)}"'


def _escape_for_gbnf_literal(text: str) -> str:
    """Escape a Python string so it is safe inside a GBNF ``"..."`` literal.

    GBNF literal syntax mirrors C string literals for our purposes: we
    need to escape backslash and double-quote.  Unicode is passed
    through as-is; the tokenizer sees raw bytes.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')
