from __future__ import annotations
import argparse
import csv
import json
import re
import tempfile
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


_THINK_BLOCK = re.compile(r"<think>.*?</think>\n*", re.DOTALL)
_FENCED_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def clean_completion(raw: str) -> str:
    """Strip <think>…</think> reasoning blocks and unwrap ```python fenced code.
    HumanEval appends the completion to the prompt and runs it as Python, so any
    prose or markdown is a syntax error. Preserves leading indentation because the
    prompt ends mid-signature and the body's indent is significant.
    """
    cleaned = _THINK_BLOCK.sub("", raw)
    fenced = _FENCED_CODE.search(cleaned)
    if fenced:
        return fenced.group(1)
    return cleaned


def run_baseline(tasks: list[dict], model: str) -> list[dict]:
    """Run each task through raw ChatOpenAI (no harness). Returns list of result dicts."""
    llm = ChatOpenAI(model=model, base_url="http://localhost:8000/v1", api_key="sk-no-key")
    results = []
    for task in tasks:
        start = time.monotonic()
        raw = llm.invoke([HumanMessage(content=task["prompt"])]).content or ""
        results.append({
            "task_id": task["task_id"],
            "completion": clean_completion(raw),
            "latency_s": round(time.monotonic() - start, 3),
            "retries": 0,
        })
    return results


def run_smboost(tasks: list[dict], model: str, scorer_threshold: float = 0.6) -> list[dict]:
    """Run each task through HarnessAgent with a CompletionTaskGraph. Returns list of result dicts."""
    from smboost import HarnessAgent, InvariantSuite
    from smboost.tasks.completion import CompletionTaskGraph

    agent = HarnessAgent(
        model=model,
        invariants=InvariantSuite.completion(),
        fallback_chain=[model],
        scorer_threshold=scorer_threshold,
        task_graph=CompletionTaskGraph(),
    )
    results = []
    for task in tasks:
        result = agent.run(task["prompt"])
        generate_output = ""
        for step in result.trace:
            if step.node == "generate":
                generate_output = step.output
        results.append({
            "task_id": task["task_id"],
            "completion": clean_completion(generate_output),
            "latency_s": result.stats.total_latency_s,
            "retries": result.stats.retry_count,
        })
    return results


def evaluate_pass_at_1(results: list[dict]) -> float:
    """Write results to temp JSONL and run HumanEval evaluation. Returns pass@1 float."""
    from human_eval.evaluation import evaluate_functional_correctness

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in results:
            json.dump({"task_id": r["task_id"], "completion": r["completion"]}, f)
            f.write("\n")
        tmp_path = f.name

    scores = evaluate_functional_correctness(tmp_path, k=[1], ignore_incomplete=True)
    return scores["pass@1"]


def write_csv(out_path: str, model: str, mode: str, pass_at_1: float,
              avg_retries: float | None, avg_latency_s: float) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "mode", "pass@1", "avg_retries", "avg_latency_s"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model": model,
            "mode": mode,
            "pass@1": round(pass_at_1, 4),
            "avg_retries": round(avg_retries, 2) if avg_retries is not None else "-",
            "avg_latency_s": round(avg_latency_s, 2),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SMBoost vs baseline on HumanEval")
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--mode", choices=["baseline", "smboost", "upper_bound"], default="baseline")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--out", default="benchmarks/results/results.csv")
    parser.add_argument("--scorer-threshold", type=float, default=0.6)
    args = parser.parse_args()

    from benchmarks.tasks import load_humaneval_tasks
    tasks = load_humaneval_tasks(args.n_tasks)
    print(f"Loaded {len(tasks)} tasks")

    if args.mode == "smboost":
        results = run_smboost(tasks, args.model, scorer_threshold=args.scorer_threshold)
    else:
        results = run_baseline(tasks, args.model)

    pass_at_1 = evaluate_pass_at_1(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    avg_retries = (
        sum(r["retries"] for r in results) / len(results) if args.mode == "smboost" else None
    )

    write_csv(args.out, args.model, args.mode, pass_at_1, avg_retries, avg_latency)

    print(
        f"{args.model} | {args.mode} | pass@1={pass_at_1:.2%} "
        f"| avg_latency={avg_latency:.1f}s"
        + (f" | avg_retries={avg_retries:.1f}" if avg_retries is not None else "")
    )


if __name__ == "__main__":
    main()
