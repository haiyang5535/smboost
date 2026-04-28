# SMBoost Harness

> A LangGraph reliability harness that lifts a 2B open-weight model on a MacBook from 20 % → 68 % on GSM8K (math) and 44 % → 74 % on HumanEval+ (code). No fine-tuning, no cloud — just better decoding.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Headline

Two benchmarks, one harness, n = 50 per cell, all on a MacBook:

| Model | Bench | n | Raw | + SMBoost (C5) | Lift |
|---|---|---|---|---|---|
| Qwen 2.5 2B (Q4_K_M, local) | GSM8K | 50 | 20.0% | **68.0%** | **3.40×** (+48pp) |
| Qwen 2.5 2B (Q4_K_M, local) | HumanEval+ | 50 | 44.0% | **74.0%** | **1.68×** (+30pp) |
| Qwen 2.5 0.8B (Q4_K_M, local) | GSM8K | 50 | 50.0% | 58.0% | 1.16× (+8pp) |

Per-task data lives at `benchmarks/results/full_matrix_v2.csv` (GSM8K)
and `benchmarks/results/he_n50_2b.csv` (HumanEval+) once you reproduce
the runs locally. The directory is `.gitignored`; the Quickstart below
regenerates it from a clean clone.

Of the 40 GSM8K problems Qwen 2B raw gets wrong, the harness recovers 25 with **zero regressions** on the 10 it already gets right. Same zero-regression signature on HumanEval+: **0 raw-only**, **15 new recoveries** at n = 50.

## Why this matters

Production teams pay for 70B-class API tokens to get reliable structured output. SMBoost shows that a 2B model with the right decoding harness — parallel sampling + program-execution verifier + raw-anchored majority vote — closes most of that reliability gap on math reasoning, locally, at near-zero cost per query.

## Quickstart

Tested on macOS / Python 3.10+.

```bash
# 1. Clone and install
git clone https://github.com/haiyang5535/smboost && cd smboost
pip install -e ".[bench,local]"

# 2. Download Qwen 2.5 2B GGUF and start the llama.cpp server
mkdir -p models
wget -O models/Qwen3.5-2B-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf"

python3 -m llama_cpp.server \
  --model models/Qwen3.5-2B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8000 \
  --n_gpu_layers -1 --n_ctx 8192 \
  --chat_format qwen \
  --chat_template_kwargs '{"enable_thinking": false}' &

# 3. Reproduce a 20-task slice of the headline
export SMBOOST_LLM_BACKEND=server
export SMBOOST_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export SMBOOST_OPENAI_API_KEY=sk-no-key

python3 scripts/run_gates_0_8b.py \
  --stage GSM8K --model qwen3.5:2b \
  --out-csv /tmp/quickstart.csv \
  --n 20 --modes raw,C5
```

Expected: raw ≈ 20%, C5 ≈ 60-75%. Full n=50 takes ~30 min on an M2 Pro.

## How it works

The headline result uses **C5 = self-consistency**, wired in [`src/smboost/harness/self_consistency_graph.py`](src/smboost/harness/self_consistency_graph.py):

1. Sample 5 completions in parallel — sample 0 at `temperature=0` (deterministic raw anchor), samples 1–4 at `temperature=0.7` (diversity).
2. Run each through a per-benchmark verifier:
   - `execute_program_verifier` for math (extracts `#### N` or runs generated Python)
   - `run_tests_verifier` for code (runs `prompt + completion + tests + check(entry_point)`)
3. Majority-vote among verifier-valid samples. Tie-break to sample 0 so no-consensus runs fall back to the deterministic baseline (no drift).

Other conditions (C1 grounded retry, C2 AST-only, C4 invariant suite, C6 real tools) are ablations included in the matrix CSV.

## Reproducing the full matrix

```bash
python3 scripts/run_full_matrix_v2.py \
  --benches gsm8k --n 50 \
  --modes raw,C5 \
  --models qwen3.5:0.8b,qwen3.5:2b
```

Crash-safe: re-running skips cells already present in the output CSV.

Gate report (G1a–G5a) regenerates from CSV with `--report-only`.

## Running the tests

```bash
pytest tests/unit -q
```

485 unit tests cover the harness, conditions, verifiers, gates, and matrix
runner. (Integration / slow tests deselected by default.)

## What this is not

- **Not a fine-tune.** No weights are touched. Same Q4_K_M GGUF in, better answers out.
- **Not a generic agent framework.** The win is concentrated where a verifier exists (math, code with tests, structured tool calls).
- **Not a universal lift on every base model.** A boundary probe on
  Phi-3 mini-4k (already-strong on GSM8K at 85 % raw) showed C5 lifts
  collapse — sample diversity overrides correct deterministic
  samples more often than it recovers wrong ones. SMBoost is the
  right tool for *low-baseline small models pushed to their ceiling*,
  not already-saturated ones. (See [`docs/overnight/2026-04-27-final-improvement-report.md`](docs/overnight/2026-04-27-final-improvement-report.md).)

## License

Apache 2.0. See [LICENSE](LICENSE).
