from __future__ import annotations
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from benchmarks.livecodebench.conditions import CONDITIONS
from benchmarks.livecodebench.loader import load_livecodebench_tasks
from benchmarks.livecodebench.runner import HEADER, run_one_cell


SHARD_DIR = Path(os.getenv("SMBOOST_MATRIX_SHARD_DIR", "benchmarks/results/livecodebench_shards"))
MEMORY_LOG_DIR = Path(os.getenv("SMBOOST_MATRIX_MEMORY_LOG_DIR", "benchmarks/results/memory_logs"))
FINAL_DIR = Path(os.getenv("SMBOOST_MATRIX_FINAL_DIR", "benchmarks/results"))


_SAFE = re.compile(r"[^A-Za-z0-9]+")


@dataclass
class Cell:
    condition: str
    model: str
    seed: int
    max_tokens: int | None = None

    @property
    def cell_id(self) -> str:
        cell_id = f"{self.condition}__{_SAFE.sub('_', self.model)}__s{self.seed}"
        if self.max_tokens is not None:
            cell_id += f"__tok{self.max_tokens}"
        return cell_id


def plan_cells(
    conditions: list[str],
    models: list[str],
    seeds: list[int],
    max_tokens: int | None = None,
) -> list[Cell]:
    return [Cell(c, m, s, max_tokens=max_tokens) for c in conditions for m in models for s in seeds]


def _run_cell(cell: Cell, tasks: list[dict]) -> Path:
    shard = SHARD_DIR / f"{cell.cell_id}.csv"
    mem_log = MEMORY_LOG_DIR / f"{cell.cell_id}.jsonl"
    run_one_cell(
        cell.condition, CONDITIONS[cell.condition],
        tasks, model=cell.model, seed=cell.seed,
        shard_path=shard, memory_log_path=mem_log,
    )
    return shard


def _effective_concurrency(ollama_concurrency: int) -> int:
    backend = os.getenv("SMBOOST_LLM_BACKEND", "server").lower()
    if backend == "local":
        return 1
    return min(ollama_concurrency, 3)


def run_matrix(
    conditions: list[str],
    models: list[str],
    seeds: list[int],
    n_tasks: int = 50,
    ollama_concurrency: int = 2,
    max_tokens: int | None = None,
) -> Path:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_LOG_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_livecodebench_tasks(n=n_tasks)
    cells = plan_cells(conditions, models, seeds, max_tokens=max_tokens)

    with ThreadPoolExecutor(max_workers=_effective_concurrency(ollama_concurrency)) as pool:
        futures = {pool.submit(_run_cell, c, tasks): c for c in cells}
        for fut in as_completed(futures):
            cell = futures[fut]
            try:
                fut.result()
                print(f"[ok] {cell.cell_id}")
            except Exception as exc:
                print(f"[ERR] {cell.cell_id}: {exc}")

    return _merge_shards()


def _merge_shards() -> Path:
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = FINAL_DIR / f"livecodebench_hard_matrix_{ts}.csv"
    shards = sorted(SHARD_DIR.glob("*.csv"))
    with open(out, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=HEADER)
        writer.writeheader()
        for s in shards:
            with open(s) as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    writer.writerow(row)
    return out
