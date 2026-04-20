import json
from pathlib import Path

def load_livecodebench_tasks(n: int | None = None) -> list[dict]:
    """
    Load the LiveCodeBench Hard subset (50 problems) from the cached jsonl.
    Record schema: {task_id, prompt, test_code, entry_point, difficulty, source,
                    testtype, test_cases}
    testtype and test_cases are extracted from the test_code JSON blob.
    """
    data_path = Path(__file__).parent.parent / "data" / "livecodebench_hard_v1.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run the download script first.")

    tasks = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            task = json.loads(line)
            _enrich_task(task)
            tasks.append(task)

    if n is not None:
        tasks = tasks[:n]

    return tasks


def _enrich_task(task: dict) -> None:
    """Extract testtype and test_cases from the test_code JSON blob in-place."""
    try:
        tc_list = json.loads(task.get("test_code", "[]"))
        # First dict element carries testtype and inline test cases
        first_dict = next((x for x in tc_list if isinstance(x, dict)), {})
        task["testtype"] = first_dict.get("testtype", "stdin")
        # Collect all dict elements as test_cases (some tasks inline all cases in the first dict)
        task["test_cases"] = [x for x in tc_list if isinstance(x, dict)]
    except Exception:
        task["testtype"] = "stdin"
        task["test_cases"] = []
