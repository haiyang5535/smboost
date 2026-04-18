from unittest.mock import MagicMock, patch
from smboost.harness.graph import HarnessGraph
from smboost.harness.state import HarnessState, StepOutput
from smboost.invariants.suite import InvariantSuite


def _make_state(**overrides) -> HarnessState:
    base: HarnessState = {
        "task": "test task",
        "model": "qwen3.5:2b",
        "fallback_chain": ["qwen3.5:2b", "qwen3.5:8b"],
        "step_outputs": [],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 0,
        "shrinkage_level": 0,
        "status": "running",
        "final_output": None,
    }
    return {**base, **overrides}


def _make_task_graph(node_names=None, output="mock output"):
    tg = MagicMock()
    tg.node_names = node_names or ["plan"]
    tg.get_node_fn.return_value = lambda state, llm: output
    return tg


def _pass_suite(node_names=None):
    names = node_names or ["plan"]
    return InvariantSuite({n: ([], [lambda s, o: True]) for n in names})


def _fail_suite(node_names=None):
    names = node_names or ["plan"]
    return InvariantSuite({n: ([], [lambda s, o: False]) for n in names})


def test_successful_single_node_sets_status_success():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _pass_suite(["plan"]), max_retries=3)

    with patch("smboost.harness.graph.ChatOllama"):
        result = graph._execute_step(_make_state())

    assert result["status"] == "success"
    assert result["step_outputs"][-1].passed is True


def test_failed_exit_invariant_increments_retry_count():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _fail_suite(["plan"]), max_retries=3)

    with patch("smboost.harness.graph.ChatOllama"):
        result = graph._execute_step(_make_state())

    assert result["retry_count"] == 1
    assert result["step_outputs"][-1].passed is False
    assert result["status"] == "running"


def test_retry_exhaustion_triggers_fallback():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _fail_suite(["plan"]), max_retries=2)

    with patch("smboost.harness.graph.ChatOllama"):
        result = graph._execute_step(_make_state(retry_count=2, fallback_index=0))

    assert result["model"] == "qwen3.5:8b"
    assert result["fallback_index"] == 1
    assert result["retry_count"] == 1  # reset to 0, then +1 for this failure


def test_fallback_exhaustion_sets_status_failed():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _fail_suite(["plan"]), max_retries=2)

    with patch("smboost.harness.graph.ChatOllama"):
        result = graph._execute_step(
            _make_state(retry_count=2, fallback_index=1,
                        model="qwen3.5:8b",
                        fallback_chain=["qwen3.5:2b", "qwen3.5:8b"])
        )

    assert result["status"] == "failed"


def test_route_returns_end_on_success():
    from langgraph.graph import END
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _pass_suite(["plan"]), max_retries=3)

    route = graph._route(_make_state(status="success"))
    assert route == END


def test_route_returns_end_on_failed():
    from langgraph.graph import END
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _pass_suite(["plan"]), max_retries=3)

    route = graph._route(_make_state(status="failed"))
    assert route == END


def test_route_continues_when_running():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _pass_suite(["plan"]), max_retries=3)

    route = graph._route(_make_state(status="running"))
    assert route == "execute_step"


def test_advances_node_index_on_success_with_multiple_nodes():
    tg = _make_task_graph(["plan", "execute"])
    suite = InvariantSuite({
        "plan": ([], [lambda s, o: True]),
        "execute": ([], [lambda s, o: True]),
    })
    graph = HarnessGraph(tg, suite, max_retries=3)

    with patch("smboost.harness.graph.ChatOllama"):
        result = graph._execute_step(_make_state(current_node_index=0))

    assert result["current_node_index"] == 1
    assert result["status"] == "running"  # still more nodes
