from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmarks.livecodebench.frontier_runner import run_frontier
from benchmarks.livecodebench.frontier import FrontierOutput


def test_frontier_runner_writes_csv(tmp_path):
    tasks = [{"task_id": "lcb_1", "prompt": "p", "test_code": "[]",
              "entry_point": "double"}]

    client = MagicMock()
    out_obj = FrontierOutput(text="def double(x):\n    return x * 2",
                             input_tokens=10, output_tokens=5, latency_ms=100)
    client.generate.return_value = out_obj

    csv_path = tmp_path / "frontier.csv"
    with patch("benchmarks.livecodebench.frontier_runner.calc_cost", return_value=0.001):
        with patch("benchmarks.livecodebench.frontier_runner.sandbox_run",
                   return_value={"passed": True, "traceback": ""}):
            run_frontier(
                system_name="gpt-4o", client=client, model_name="gpt-4o",
                tasks=tasks, csv_path=csv_path,
            )
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert "gpt-4o" in lines[1]
    assert ",1," in lines[1]  # passed
