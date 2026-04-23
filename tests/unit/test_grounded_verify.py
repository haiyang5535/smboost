from unittest.mock import MagicMock
from smboost.harness.state import HarnessState, StepOutput
from smboost.tasks.completion import CompletionTaskGraph


def _state(solution: str, task_metadata: dict) -> HarnessState:
    return {
        "task": "prompt",
        "task_metadata": task_metadata,
        "model": "x",
        "fallback_chain": ["x"],
        "step_outputs": [StepOutput(node="generate", model="x", output=solution, confidence=1.0, passed=True)],
        "retry_count": 0,
        "fallback_index": 0,
        "current_node_index": 1,
        "shrinkage_level": 0,
        "status": "running",
        "final_output": solution,
    }


def test_verify_node_grounded_pass():
    graph = CompletionTaskGraph(grounded_verify=True)
    state = _state(
        "def double(x):\n    return x * 2",
        {"testtype": "functional", "entry_point": "double",
         "test_cases": [{"input": "2", "output": "4"}]},
    )
    result = graph.get_node_fn("verify")(state, MagicMock())
    assert result == "PASS"


def test_verify_node_grounded_fail_with_traceback():
    graph = CompletionTaskGraph(grounded_verify=True)
    state = _state(
        "def double(x):\n    return x + 1",
        {"testtype": "functional", "entry_point": "double",
         "test_cases": [{"input": "2", "output": "4"}]},
    )
    result = graph.get_node_fn("verify")(state, MagicMock())
    assert result.startswith("FAIL")
    assert "assert" in result.lower() or "error" in result.lower()


def test_verify_node_grounded_stdin_wraps_top_level_solve():
    graph = CompletionTaskGraph(grounded_verify=True)
    state = _state(
        "import sys\n\n"
        "def solve():\n"
        "    print(int(sys.stdin.read().strip()) + 1)\n",
        {"testtype": "stdin", "test_cases": [{"input": "5\n", "output": "6\n"}]},
    )
    result = graph.get_node_fn("verify")(state, MagicMock())
    assert result == "PASS"


def test_verify_node_grounded_stdin_keeps_existing_solve_call():
    graph = CompletionTaskGraph(grounded_verify=True)
    state = _state(
        "import sys\n\n"
        "def solve():\n"
        "    print(int(sys.stdin.read().strip()) + 1)\n\n"
        "if __name__ == '__main__':\n"
        "    solve()\n",
        {"testtype": "stdin", "test_cases": [{"input": "5\n", "output": "6\n"}]},
    )
    result = graph.get_node_fn("verify")(state, MagicMock())
    assert result == "PASS"


def test_verify_node_grounded_stdin_skeleton_program_passes():
    graph = CompletionTaskGraph(grounded_verify=True)
    state = _state(
        "import sys\n\n"
        "def solve():\n"
        "    data = sys.stdin.read().strip().split()\n"
        "    sys.stdout.write(str(int(data[0]) + 1) + '\\n')\n",
        {"testtype": "stdin", "test_cases": [{"input": "5\n", "output": "6\n"}]},
    )
    result = graph.get_node_fn("verify")(state, MagicMock())
    assert result == "PASS"


def test_verify_node_ast_only_when_grounded_off():
    graph = CompletionTaskGraph(grounded_verify=False)
    state = _state(
        "def double(x):\n    return x * 99",  # wrong logic, valid syntax
        {"testtype": "functional", "entry_point": "double",
         "test_cases": [{"input": "2", "output": "4"}]},
    )
    result = graph.get_node_fn("verify")(state, MagicMock())
    assert result == "PASS"  # ast-only can't detect wrong logic


def test_verify_node_falls_back_to_ast_when_no_metadata():
    graph = CompletionTaskGraph(grounded_verify=True)
    state = _state("def double(x):\n    return x * 2", {})
    result = graph.get_node_fn("verify")(state, MagicMock())
    assert result == "PASS"  # ast-only fallback when no testtype


def test_verify_node_detects_syntax_error():
    graph = CompletionTaskGraph(grounded_verify=True)
    state = _state("def double(x\n    return", {})
    result = graph.get_node_fn("verify")(state, MagicMock())
    assert result.startswith("FAIL")
    assert "syntax" in result.lower()
