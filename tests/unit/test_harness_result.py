from smboost.harness.result import HarnessResult, RunStats
from smboost.harness.state import StepOutput

def test_run_stats_fields():
    stats = RunStats(retry_count=2, fallback_triggers=1, total_latency_s=1.23, model_used="qwen3.5:2b")
    assert stats.retry_count == 2
    assert stats.model_used == "qwen3.5:2b"

def test_harness_result_fields():
    step = StepOutput(node="plan", model="qwen3.5:2b", output="ok", confidence=1.0, passed=True)
    stats = RunStats(retry_count=0, fallback_triggers=0, total_latency_s=0.5, model_used="qwen3.5:2b")
    result = HarnessResult(output="done", trace=[step], stats=stats, status="success")
    assert result.output == "done"
    assert len(result.trace) == 1
    assert result.status == "success"
