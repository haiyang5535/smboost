# tests/unit/test_scorer.py
from unittest.mock import MagicMock
from smboost.scorer import RobustnessScorer


def _make_state():
    return {
        "task": "test",
        "model": "qwen3.5:2b",
        "fallback_chain": ["qwen3.5:2b"],
        "step_outputs": [],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 0,
        "shrinkage_level": 0,
        "status": "running",
        "final_output": None,
    }


def test_identical_outputs_give_confidence_1():
    scorer = RobustnessScorer(n_samples=3)
    node_fn = MagicMock(return_value="same output")

    output, confidence = scorer.score(node_fn, _make_state(), MagicMock())

    assert confidence == 1.0
    assert output == "same output"
    assert node_fn.call_count == 3


def test_completely_different_outputs_give_low_confidence():
    scorer = RobustnessScorer(n_samples=3)
    outputs_iter = iter(["aaa bbb ccc", "xyz 123 789", "foo bar qux"])
    node_fn = MagicMock(side_effect=lambda s, l: next(outputs_iter))

    _, confidence = scorer.score(node_fn, _make_state(), MagicMock())

    assert confidence < 0.3


def test_two_matching_one_outlier_returns_matching_as_centroid():
    scorer = RobustnessScorer(n_samples=3)
    outputs_iter = iter(["hello world", "hello world", "zzzzz"])
    node_fn = MagicMock(side_effect=lambda s, l: next(outputs_iter))

    output, confidence = scorer.score(node_fn, _make_state(), MagicMock())

    assert output == "hello world"
    assert 0.3 < confidence < 1.0


def test_single_sample_returns_confidence_1():
    scorer = RobustnessScorer(n_samples=1)
    node_fn = MagicMock(return_value="only output")

    output, confidence = scorer.score(node_fn, _make_state(), MagicMock())

    assert confidence == 1.0
    assert output == "only output"


def test_threshold_stored_on_instance():
    scorer = RobustnessScorer(threshold=0.5)
    assert scorer.threshold == 0.5


def test_scorer_passes_state_and_llm_to_node_fn():
    scorer = RobustnessScorer(n_samples=2)
    node_fn = MagicMock(return_value="output")
    state = _make_state()
    llm = MagicMock()

    scorer.score(node_fn, state, llm)

    node_fn.assert_called_with(state, llm)
