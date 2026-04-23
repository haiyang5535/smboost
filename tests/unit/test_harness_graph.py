from unittest.mock import MagicMock, patch
from smboost.harness.graph import HarnessGraph
from smboost.harness.state import HarnessState, StepOutput
from smboost.invariants.suite import InvariantSuite


def _make_state(**overrides) -> HarnessState:
    base: HarnessState = {
        "task": "test task",
        "task_metadata": {},
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


def _noop_factory(model: str):
    """LLM factory that returns a MagicMock — good enough for unit tests
    where the node_fn ignores the llm argument."""
    return MagicMock()


def test_successful_single_node_sets_status_success():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _pass_suite(["plan"]), max_retries=3, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(_make_state())

    assert result["status"] == "success"
    assert result["step_outputs"][-1].passed is True


def test_failed_exit_invariant_increments_retry_count():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _fail_suite(["plan"]), max_retries=3, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(_make_state())

    assert result["retry_count"] == 1
    assert result["step_outputs"][-1].passed is False
    assert result["status"] == "running"


def test_retry_exhaustion_triggers_fallback():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _fail_suite(["plan"]), max_retries=2, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(_make_state(retry_count=2, fallback_index=0))

    assert result["model"] == "qwen3.5:8b"
    assert result["fallback_index"] == 1
    assert result["retry_count"] == 1  # reset to 0, then +1 for this failure


def test_fallback_exhaustion_sets_status_failed():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _fail_suite(["plan"]), max_retries=2, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(
            _make_state(retry_count=2, fallback_index=1,
                        model="qwen3.5:8b",
                        fallback_chain=["qwen3.5:2b", "qwen3.5:8b"])
        )

    assert result["status"] == "failed"


def test_route_returns_end_on_success():
    from langgraph.graph import END
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _pass_suite(["plan"]), max_retries=3, llm_factory=_noop_factory)

    route = graph._route(_make_state(status="success"))
    assert route == END


def test_route_returns_end_on_failed():
    from langgraph.graph import END
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _pass_suite(["plan"]), max_retries=3, llm_factory=_noop_factory)

    route = graph._route(_make_state(status="failed"))
    assert route == END


def test_route_continues_when_running():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _pass_suite(["plan"]), max_retries=3, llm_factory=_noop_factory)

    route = graph._route(_make_state(status="running"))
    assert route == "execute_step"


def test_advances_node_index_on_success_with_multiple_nodes():
    tg = _make_task_graph(["plan", "execute"])
    suite = InvariantSuite({
        "plan": ([], [lambda s, o: True]),
        "execute": ([], [lambda s, o: True]),
    })
    graph = HarnessGraph(tg, suite, max_retries=3, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(_make_state(current_node_index=0))

    assert result["current_node_index"] == 1
    assert result["status"] == "running"  # still more nodes


from unittest.mock import MagicMock
from smboost.scorer import RobustnessScorer


def _make_mock_scorer(confidence: float) -> MagicMock:
    scorer = MagicMock(spec=RobustnessScorer)
    scorer.score.return_value = ("scorer output", confidence)
    scorer.threshold = 0.6
    return scorer


def test_systematic_failure_increments_shrinkage_level():
    tg = _make_task_graph(["plan"])
    scorer = _make_mock_scorer(confidence=0.2)  # below threshold
    graph = HarnessGraph(tg, _fail_suite(["plan"]), max_retries=3, scorer=scorer, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(_make_state())

    assert result["shrinkage_level"] == 1
    assert result["retry_count"] == 1
    assert result["step_outputs"][-1].confidence == 0.2


def test_transient_failure_keeps_shrinkage_level_unchanged():
    tg = _make_task_graph(["plan"])
    scorer = _make_mock_scorer(confidence=0.9)  # above threshold
    graph = HarnessGraph(tg, _fail_suite(["plan"]), max_retries=3, scorer=scorer, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(_make_state(shrinkage_level=2))

    assert result["shrinkage_level"] == 2
    assert result["retry_count"] == 1


def test_small_model_stdin_verify_bypasses_scorer_and_increments_shrinkage():
    tg = _make_task_graph(["generate", "verify"], output="FAIL: syntax")
    scorer = _make_mock_scorer(confidence=0.9)
    graph = HarnessGraph(tg, _fail_suite(["generate", "verify"]), max_retries=3, scorer=scorer, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(
            _make_state(
                current_node_index=1,
                task_metadata={"testtype": "stdin"},
                model="qwen3.5:2b",
            )
        )

    scorer.score.assert_not_called()
    assert result["shrinkage_level"] == 1
    assert result["retry_count"] == 1
    assert result["step_outputs"][-1].confidence == 0.0


def test_small_model_stdin_generate_bypasses_scorer_and_increments_shrinkage():
    tg = _make_task_graph(["generate", "verify"], output="")
    scorer = _make_mock_scorer(confidence=0.9)
    graph = HarnessGraph(tg, _fail_suite(["generate", "verify"]), max_retries=3, scorer=scorer, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(
            _make_state(
                current_node_index=0,
                task_metadata={"testtype": "stdin"},
                model="qwen3.5:2b",
            )
        )

    scorer.score.assert_not_called()
    assert result["shrinkage_level"] == 1
    assert result["retry_count"] == 1
    assert result["step_outputs"][-1].confidence == 0.0


def test_non_stdin_verify_still_uses_scorer():
    tg = _make_task_graph(["generate", "verify"], output="FAIL: syntax")
    scorer = _make_mock_scorer(confidence=0.9)
    graph = HarnessGraph(tg, _fail_suite(["generate", "verify"]), max_retries=3, scorer=scorer, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(
            _make_state(
                current_node_index=1,
                task_metadata={"testtype": "functional"},
                model="qwen3.5:2b",
                shrinkage_level=2,
            )
        )

    scorer.score.assert_called_once()
    assert result["shrinkage_level"] == 2
    assert result["retry_count"] == 1


def test_shrinkage_exhaustion_triggers_early_fallback():
    tg = _make_task_graph(["plan"])
    scorer = _make_mock_scorer(confidence=0.1)  # systematic
    graph = HarnessGraph(tg, _fail_suite(["plan"]), max_retries=5, scorer=scorer, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(_make_state(shrinkage_level=3, fallback_index=0))

    assert result["model"] == "qwen3.5:8b"
    assert result["fallback_index"] == 1
    assert result["shrinkage_level"] == 0


def test_shrinkage_exhaustion_with_no_fallback_sets_failed():
    tg = _make_task_graph(["plan"])
    scorer = _make_mock_scorer(confidence=0.1)
    graph = HarnessGraph(tg, _fail_suite(["plan"]), max_retries=5, scorer=scorer, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(
            _make_state(shrinkage_level=3, fallback_index=1,
                        model="qwen3.5:8b",
                        fallback_chain=["qwen3.5:2b", "qwen3.5:8b"])
        )

    assert result["status"] == "failed"


def test_success_resets_shrinkage_level_to_zero():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _pass_suite(["plan"]), max_retries=3, llm_factory=_noop_factory)

    with patch.object(graph, "_llm_factory", lambda model: MagicMock()):
        result = graph._execute_step(_make_state(shrinkage_level=2))

    assert result["shrinkage_level"] == 0


def test_default_scorer_is_none():
    tg = _make_task_graph(["plan"])
    graph = HarnessGraph(tg, _pass_suite(["plan"]), llm_factory=_noop_factory)
    assert graph._scorer is None


def test_explicit_scorer_is_stored():
    tg = _make_task_graph(["plan"])
    scorer = RobustnessScorer()
    graph = HarnessGraph(tg, _pass_suite(["plan"]), scorer=scorer, llm_factory=_noop_factory)
    assert isinstance(graph._scorer, RobustnessScorer)
