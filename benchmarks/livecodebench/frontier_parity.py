import argparse
import os
import csv
import time
from pathlib import Path
from benchmarks.livecodebench.loader import load_livecodebench_tasks
from benchmarks.livecodebench.sandbox import run
from benchmarks.livecodebench.frontier import get_frontier_model

def main():
    parser = argparse.ArgumentParser(description="Run LiveCodeBench on Frontier Models for Cost Parity")
    parser.add_argument("--models", nargs="+", required=True, help="Frontier models to evaluate (e.g. gpt-4o, claude-3-5-sonnet-20240620)")
    parser.add_argument("--dry-run", action="store_true", help="Just print what would run")
    args = parser.parse_args()

    # Pricing snapshot as of Day 12 (Apr 2026 placeholder)
    PRICING = {
        "gpt-4o": {"in": 5.0, "out": 15.0}, # per 1M tokens
        "claude-3-5-sonnet-20240620": {"in": 3.0, "out": 15.0},
        "gpt-4-turbo": {"in": 10.0, "out": 30.0},
        "claude-3-opus-20240229": {"in": 15.0, "out": 75.0}
    }

    try:
        tasks = load_livecodebench_tasks(50)
    except FileNotFoundError:
        print("Error: Dataset not found. Run loader script first.")
        return

    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"frontier_parity_{timestamp}.csv"
    
    fieldnames = [
        "system", "task_id", "passed", "total_usd", "p50_ms", "p95_ms", "avg_retries"
    ]

    print(f"Starting frontier parity evaluation on {len(tasks)} tasks.")
    print(f"Models: {args.models}")
    
    if args.dry_run:
        print("Dry run requested. Exiting.")
        return
        
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for model_name in args.models:
        print(f"Evaluating {model_name}...")
        try:
            model = get_frontier_model(model_name)
        except ValueError as e:
            print(f"Skipping {model_name}: {e}")
            continue

        pricing = PRICING.get(model_name, {"in": 0.0, "out": 0.0})
        
        for task in tasks:
            task_id = task["task_id"]
            
            # One retry on transient API errors, 0 on task failures
            max_retries = 1
            success = False
            api_latency = 0
            usd_cost = 0.0
            
            for attempt in range(max_retries + 1):
                try:
                    result = model.generate(task["prompt"])
                    
                    usage = result["usage"]
                    in_tokens = usage.get("prompt_tokens", 0)
                    out_tokens = usage.get("completion_tokens", 0)
                    
                    usd_cost = (in_tokens / 1_000_000 * pricing["in"]) + (out_tokens / 1_000_000 * pricing["out"])
                    api_latency = result["latency_ms"]
                    
                    # Clean markdown
                    code_output = result["output"]
                    if "```python" in code_output:
                        code_output = code_output.split("```python")[1].split("```")[0]
                    elif "```" in code_output:
                        code_output = code_output.split("```")[1].split("```")[0]
                    
                    sandbox_start = time.perf_counter()
                    sandbox_res = run(code_output.strip(), task["test_code"], task["entry_point"])
                    sandbox_latency = int((time.perf_counter() - sandbox_start) * 1000)
                    
                    total_latency = api_latency + sandbox_latency
                    
                    with open(csv_path, "a", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow({
                            "system": model_name,
                            "task_id": task_id,
                            "passed": sandbox_res["passed"],
                            "total_usd": usd_cost,
                            "p50_ms": total_latency, # We'll aggregate these in the final report
                            "p95_ms": total_latency,
                            "avg_retries": attempt
                        })
                    
                    success = True
                    break # Task completed, exit retry loop
                    
                except Exception as e:
                    print(f"Transient error on {model_name} task {task_id}: {e}")
                    time.sleep(2)
                    
            if not success:
                print(f"Failed to evaluate {task_id} with {model_name} after retries.")
                with open(csv_path, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({
                        "system": model_name,
                        "task_id": task_id,
                        "passed": False,
                        "total_usd": 0.0,
                        "p50_ms": 0,
                        "p95_ms": 0,
                        "avg_retries": max_retries
                    })

    print(f"Frontier evaluation complete. Results saved to {csv_path}")

if __name__ == "__main__":
    main()
