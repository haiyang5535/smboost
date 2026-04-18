# SMBoost Harness — State Machine Design

**Date:** 2026-04-16 (Day 1)
**Scope:** Core harness state machine + two Day 1 demos (coding agent, tool-calling agent)

---

## Problem

Raw Qwen 3.5 2B (and 2B-class models generally) fail in production because they have no structured retry, fallback, or invariant enforcement — not because they lack knowledge. This design spec covers the foundational state machine that gives the model a reliability layer without touching weights.

---

## Architecture

Two-level graph structure:

```
HarnessAgent (public API entry point)
│
├── HarnessGraph (outer LangGraph StateGraph)
│   ├── Owns: retry budget, fallback_chain, run stats, failure memory (session)
│   ├── Wraps every inner node call: entry invariant → run → exit invariant → retry/escalate
│   └── Produces: HarnessResult(output, trace, stats)
│
└── TaskGraph (inner, swappable per task type)
    ├── CodingTaskGraph         — demo 1
    │   nodes: plan → execute → verify
    │   tools: read_file, write_file, run_shell
    │
    └── ToolCallingTaskGraph    — demo 2
        nodes: plan → dispatch → verify
        tools: configurable at init time
```

The outer `HarnessGraph` is responsible for all harness concerns (retry, fallback, invariant checking, stats). The inner `TaskGraph` is responsible only for task-domain logic. This separation means harness behavior is never duplicated across nodes.

---

## Package Layout

```
src/smboost/
├── __init__.py              # exports: HarnessAgent, InvariantSuite
├── agent.py                 # HarnessAgent — public entry point
├── harness/
│   ├── graph.py             # HarnessGraph (outer StateGraph)
│   ├── state.py             # HarnessState, StepOutput typed dataclasses
│   └── result.py            # HarnessResult, RunStats
├── tasks/
│   ├── base.py              # TaskGraph ABC
│   ├── coding.py            # CodingTaskGraph
│   └── tool_calling.py      # ToolCallingTaskGraph
└── invariants/
    └── suite.py             # InvariantSuite + built-in invariant fns

tests/
├── unit/
│   ├── test_invariants.py
│   ├── test_harness_graph.py
│   └── test_harness_result.py
└── integration/
    └── test_agent_e2e.py    # requires Ollama + qwen3.5:2b running
```

---

## State

```python
@dataclass
class HarnessState:
    task:           str
    model:          str                    # active model; changes on fallback
    fallback_chain: list[str]
    step_outputs:   list[StepOutput]
    retry_count:    int
    fallback_index: int
    entry_ok:       bool
    exit_ok:        bool
    status:         Literal["running", "success", "failed"]
    final_output:   str | None

@dataclass
class StepOutput:
    node:       str
    model:      str
    output:     str
    confidence: float   # 1.0 for MVP; wired to adaptive scorer in Day 3-5
    passed:     bool
```

---

## Data Flow (per node execution)

```
HarnessGraph calls inner node
  1. run entry_invariants(state)  → fail → retry or escalate
  2. call ChatOllama with tools
  3. run exit_invariants(state + output) → fail → retry or escalate
  4. append StepOutput to state.step_outputs
  5. route: next_node | retry | fallback | terminal_fail
```

---

## Retry + Fallback Policy

- **Per node**, retry same model up to `max_retries` (default: 3)
- On retry exhaustion → increment `fallback_index`, reset `retry_count`, replay node with next model in `fallback_chain`
- On `fallback_chain` exhaustion → `status = "failed"`, return `HarnessResult` with failure trace
- The active model (`state.model`) is mutated in-place on fallback so the trace reflects which model ran each step

---

## Invariants

```python
# output is None for entry invariants, the node's raw output str for exit invariants
InvariantFn = Callable[[HarnessState, str | None], bool]

class InvariantSuite:
    node_invariants: dict[str, tuple[list[InvariantFn], list[InvariantFn]]]
    # key: node name
    # value: (entry_invariants, exit_invariants)

    @staticmethod
    def coding_agent() -> "InvariantSuite": ...

    @staticmethod
    def tool_calling() -> "InvariantSuite": ...
```

**Built-in invariants (Day 1):**

| Name | Type | Checks |
|---|---|---|
| `output_is_nonempty` | exit | `len(output.strip()) > 0` |
| `output_is_valid_json` | exit | `json.loads(output)` succeeds |
| `no_error_keywords` | exit | output does not contain `"Error:"`, `"Traceback"`, `"Exception:"` |

---

## Public Result Type

```python
@dataclass
class RunStats:
    retry_count:       int
    fallback_triggers: int
    total_latency_s:   float
    model_used:        str    # final model that produced the output

@dataclass
class HarnessResult:
    output: str
    trace:  list[StepOutput]
    stats:  RunStats
    status: Literal["success", "failed"]
```

---

## Model Integration

- **Backend:** `langchain_ollama.ChatOllama` (native LangGraph integration)
- **Tool binding:** LangChain tool protocol — tools defined as `@tool` functions, bound via `llm.bind_tools(tools)`
- **Default model:** `qwen3.5:2b`; fallback: `qwen3.5:8b`

---

## Testing Strategy

| Layer | Scope | Marker |
|---|---|---|
| Unit | Invariant fns in isolation | (default) |
| Unit | `HarnessGraph` routing with mock `TaskGraph` | (default) |
| Integration | Full `HarnessAgent.run()` against live Ollama | `@pytest.mark.integration` |

`pytest tests/` runs unit tests only. Integration tests require `ollama serve` with `qwen3.5:2b` pulled.

---

## Out of Scope (Day 1)

- Adaptive robustness scorer (confidence is hardcoded `1.0` for MVP)
- Shrinkage-style prompt simplification on retry
- Session-scoped failure memory
- SaaS dashboard / trace viewer
- `pip install smboost` / PyPI packaging
