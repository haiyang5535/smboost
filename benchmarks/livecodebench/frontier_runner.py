from __future__ import annotations
import csv
import re
import time
from pathlib import Path

from benchmarks.livecodebench.frontier_pricing import calc_cost
from benchmarks.livecodebench.sandbox import run as sandbox_run


HEADER = [
    "system", "task_id", "passed", "input_tokens", "output_tokens",
    "usd_cost", "latency_ms", "wall_clock_ms",
]

_FENCED_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _clean(raw: str) -> str:
    fenced = _FENCED_CODE.search(raw)
    return fenced.group(1) if fenced else raw


def run_frontier(
    *, system_name: str, client, model_name: str,
    tasks: list[dict], csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if write_header:
            writer.writeheader()
        for t in tasks:
            start = time.monotonic()
            try:
                out = client.generate(t["prompt"])
            except Exception:
                time.sleep(2)
                out = client.generate(t["prompt"])
            wall = int((time.monotonic() - start) * 1000)
            result = sandbox_run(_clean(out.text), t.get("test_code", ""), t.get("entry_point", ""))
            cost = calc_cost(model_name, out.input_tokens, out.output_tokens)
            writer.writerow({
                "system": system_name,
                "task_id": t["task_id"],
                "passed": 1 if result["passed"] else 0,
                "input_tokens": out.input_tokens,
                "output_tokens": out.output_tokens,
                "usd_cost": round(cost, 6),
                "latency_ms": out.latency_ms,
                "wall_clock_ms": wall,
            })
            f.flush()
