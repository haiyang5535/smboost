# SMBoost — Project Writeup

A long-form companion to the [README](../README.md) for readers who
want the full story: motivation, the ablation matrix and what each
condition isolates, the result tables, the mechanism behind why C5
works (and where it stops working), the failure-mode breakdown, the
debugging story behind the ground-truth evaluation bug, and references.

## Problem and motivation

A 2 B-class open-weight model (Qwen 2.5 2B, Llama 3.2 3B, Phi-3 mini)
fits in 3–6 GB of memory and runs without a network connection. That
makes it the only realistic option for a large class of deployments
(on-device assistants, edge inference, regulated environments where
data cannot leave the box). The catch is that on most reasoning
benchmarks a Q4-quantized 2 B model is too unreliable to ship: GSM8K
raw pass rate around 20 %, HumanEval+ around 44 %.

The published research on decoding-time techniques — self-consistency
(Wang et al., 2022), program / verifier-based inference (Lightman et
al., 2023; PROVE, Toh et al., 2024) — argues that most of the
reliability gap can be closed by sampling more from the same model and
keeping only the answers a cheap verifier accepts. I had not seen a
from-scratch engineering implementation of that idea applied to a Q4
quantized 2 B model on local hardware, with a clean ablation matrix.
This project is that implementation.

The deliberate scope: tasks where a *cheap, deterministic verifier*
already exists. Math word problems with a numeric answer. Code
problems with unit tests. Structured tool-call payloads with a JSON
schema. Open-ended generation is out of scope — there is no
verifier signal to majority-vote on.

## Approach: six conditions, one matrix

The harness wraps any local LLM into a LangGraph state machine and
exposes six conditions (C1–C6). Each condition isolates one
contribution to the final number. Wiring is in
`benchmarks/conditions.py::build_condition`; the per-condition flag
table is reproduced here:

| Condition | grounded_verify | session_memory | shrinkage | scorer | What it isolates |
|---|:---:|:---:|:---:|:---:|---|
| **C1** | ✅ | ✅ | ✅ | ✅ | Full grounded retry: AST + verifier loop, session memory of failures, simplified-prompt retries |
| **C2** | ❌ | ✅ | ✅ | ✅ | Drops grounded verify; retains AST-only acceptance + session memory + shrinkage |
| **C3** | ✅ | ❌ | ✅ | ✅ | Same as C1 minus session memory — measures what cross-attempt memory contributes |
| **C4** | ❌ | ❌ | ❌ | ❌ | The "no harness" floor — invariants suite only, no retry, no memory, no shrinkage |
| **C5** | ❌ | ✅ | ❌ | ✅ | Self-consistency: parallel sampling + per-sample verifier + raw-anchored majority vote |
| **C6** | ✅ | ✅ | ✅ | ❌ | C1 minus the adaptive scorer — measures whether scorer-driven retry budgeting matters |

The ablation table is the entire experimental design. The numbers
reported below are taken from these six conditions × {GSM8K,
HumanEval+} × {Qwen 2.5 2B Q4_K_M, Qwen 2.5 0.8B Q4_K_M, Phi-3 mini
4k Q4} at n = 50 (n = 20 for the boundary probe).

## Results

### Headline matrix (n = 50 per cell)

| Model | Bench | raw | C1 | **C5** | Lift (C5 / raw) |
|---|---|---|---|---|---|
| Qwen 2.5 2B | GSM8K | 20.0 % | _skipped_ ¹ | **68.0 %** | **3.40 ×** |
| Qwen 2.5 2B | HumanEval+ | 44.0 % | — | **74.0 %** | **1.68 ×** |
| Qwen 2.5 0.8B | GSM8K | 50.0 % | 56.0 % | 58.0 % | 1.16 × |

¹ 2 B C1 was skipped because n = 20 pilots showed a regression to
~10 % caused by an interaction between the adaptive scorer and the
grounded-verify retry. The bug is documented in the open-engineering
section below; C5 is the production path.

### Boundary probe (n = 20)

| Model | Bench | raw | C5 | Lift |
|---|---|---|---|---|
| Phi-3 mini-4k Q4 | GSM8K | 85.0 % | 75.0 % | **0.88 ×** (regression) |

### Zero-regression signature

Across both Qwen 2 B benchmarks the harness has the property that
*every task raw decoding gets right, C5 also gets right.* The lift
comes entirely from the "raw was wrong but a sibling sample was
right" tail.

| Cell | raw correct | C5 correct | raw\C5 | C5\raw |
|---|:---:|:---:|:---:|:---:|
| 2 B GSM8K | 10 / 50 | 34 / 50 | **0** | 25 |
| 2 B HumanEval+ | 22 / 50 | 37 / 50 | **0** | 15 |
| Phi-3 GSM8K | 17 / 20 | 15 / 20 | 2 | 0 |

The first two rows are the result the project is built around. The
third row is the boundary case discussed below.

## Why C5 works (mechanism)

C5 has three pieces, all visible in
`src/smboost/harness/self_consistency_graph.py`:

1. **Parallel sampling with a deterministic anchor.** Sample 0 is
   drawn at `temperature = 0` — this is bit-equivalent to raw
   decoding. Samples 1–4 are drawn at `temperature = 0.7` for
   diversity. The anchor is what gives the harness its zero-regression
   property: in the worst case (no consensus among the diverse
   samples), the harness *cannot* select an answer worse than what
   the deterministic baseline would have produced, because it falls
   back to the anchor.

2. **Per-sample program verifier.** Each candidate completion runs
   through a verifier picked by the benchmark — `execute_program_verifier`
   for math (extract `#### N` or run the generated Python),
   `run_tests_verifier` for code (run `prompt + completion + tests +
   check(entry_point)` in a sandboxed subprocess). The verifier
   returns a `valid` bit and an `extracted_answer` used for grouping.
   Verifier-invalid samples are excluded from the vote.

3. **Majority vote with raw-anchor tie-break.** The verifier-valid
   samples are bucketed by `extracted_answer`. The largest bucket
   wins. On no-consensus (multiple buckets tied at the top, or zero
   verifier-valid samples), the harness falls back to sample 0.

The interaction of (1) and (3) is the key engineering choice. A naive
self-consistency implementation that ties to sample 0 *only when there
is a single answer string* — but selects sample 0 from a sampled
distribution rather than pinning it — will drift on no-consensus runs.
Pinning the anchor at temperature 0 turns the failure mode into "fall
back to deterministic output." The numbers above are downstream of
this property.

## Why C5 stops working: the Phi-3 boundary

The Phi-3 mini-4k probe at n = 20 produced a regression: raw 85 %, C5
75 %. Two raw-correct tasks (`gsm8k/2`, `gsm8k/12`) drifted to wrong
under C5; zero new tasks were recovered.

The mechanism is straightforward once stated. With raw at 85 % there
are only ~3 wrong-on-raw tasks out of 20. The four `temperature = 0.7`
sibling samples have many more chances to *out-vote* sample 0 with a
confidently-consistent-but-wrong majority on tasks where the model is
already correct deterministically, than they have to *recover* a
raw-wrong task. Self-consistency, by construction, requires a
non-trivial probability that the base model produces a correct answer
*at temperature 0.7* — and on tasks the model is decisively right
about, that probability is below 1, so the diverse samples can vote
the correct sample 0 down.

The takeaway: self-consistency is a **low-baseline tool**. It is the
right harness for "small model with a lot of headroom" deployments,
not for "already-strong base model" deployments. The 1.5× pilot gate
threshold I used is the operational floor — under that, do not promote
the run.

## Failure-mode breakdown

The runner records a `failure_bucket` for every wrong answer. Reading
across the GSM8K 2 B C5 cell on the failures the harness still has
(34 / 50 pass means 16 fails):

- `wrong_answer` — the verifier said valid but the answer was numerically wrong (the dominant bucket — confidently-consistent-but-wrong)
- `no_numeric_answer` — the model emitted no `#### N` and no Python that could be executed; the verifier marked all 5 samples invalid; raw-anchor fallback then also produced an invalid output
- `syntax_truncation` — generation hit the token cap mid-CoT (rare at `max_tokens = 512` on GSM8K, but does happen)
- `other_runtime` — the program-execution verifier raised on a generated program (`ZeroDivisionError`, `IndexError`, etc.)

The `wrong_answer` bucket is the one self-consistency is structurally
unable to fix: when 4 / 5 samples agree on the wrong answer, the
majority vote ratifies it. This is the same mechanism that produces
the Phi-3 regression at higher raw baselines.

## The ground-truth evaluation bug

This is the kind of bug that is invisible until you specifically look
for it. Worth the section because it changed every conclusion the
project had made up to that point.

**What I believed (early):** the LiveCodeBench Hard runner was
returning genuine pass rates per condition. C2 and C4 (cheap
conditions, no grounded verify) were apparently outperforming C1 (the
full grounded-retry harness) on n = 150, by 5–10 pp. I spent two
overnight runs trying to understand why a *weaker* harness was
beating a *stronger* one.

**What I changed:** I traced the runner's pass accounting and found
that for C2 and C4, the runner was treating the harness's
`status="success"` as a benchmark pass. `status="success"` only means
"the AST parsed and the invariants held" — for C2 / C4 with grounded
verify *disabled*, that frequently flagged True for outputs that did
not actually pass the LiveCodeBench test cases. The C1 path, which
*did* run the test cases inside grounded verify, was being held to a
strictly tighter bar.

**The fix:** `benchmarks/livecodebench/runner.py` now executes the
final generated program through the LiveCodeBench sandbox and uses
*that* pass bit as the row's `passed` value, regardless of what the
harness reported. After the change, the C2 / C4 advantage largely
disappeared on the representative subset.

**What I believe now:** end-to-end ground-truth evaluation has to be
the gate. Anything the harness reports about itself is part of the
system under test, not part of the test. If the harness's self-report
matched the sandbox's verdict it would be redundant; if it does not
match, you cannot trust either side without running the sandbox
independently. This is why the matrix runner now duplicates work
(generate inside the harness, then re-execute outside it).

The episode is what motivates the Quickstart's choice to use real
benchmark data (`run_full_matrix_v2.py` against `humaneval_plus` and
`gsm8k`) rather than a faster synthetic eval. A faster eval that
trusts the harness would give faster wrong answers.

## Open engineering items (technical, not strategic)

- **2 B C1 regression on GSM8K.** The full grounded-retry harness
  (C1) regresses on GSM8K when paired with the adaptive scorer's
  step-accounting. The interaction is at `harness/graph.py:102` (the
  scorer-on-verify step). A 3-day debugging effort, deferred — C5 is
  the production path, so this is not blocking.
- **BFCL is structurally blocked.** `ToolCallingTaskGraph` executes
  tools via `run_tool_loop`, but BFCL "tools" are JSON schemas with
  no implementations. A proper fix needs an
  `EmitOnlyToolCallingTaskGraph` that predicts the call without
  executing it. Drop BFCL from the headline until that lands.
- **Sample count.** `n_samples = 5` is hard-coded in
  `build_condition`. Lifting to 7 or 11 is a one-line change and may
  add 2–5 pp on 2 B at proportional latency cost.
- **Verifier ensembles.** Adding a *consistency* verifier ("does the
  answer plug back into the problem?") to the GSM8K verifier picks up
  a class of confidently-consistent-but-wrong majority votes that
  pure execution verification cannot catch.

## References

- *Self-Consistency Improves Chain of Thought Reasoning in Language
  Models* — Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou
  (2022). [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- *Let's Verify Step by Step* — Lightman, Kosaraju, Burda, Edwards,
  Baker, Lee, Leike, Schulman, Sutskever, Cobbe (2023).
  [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- *PROVE: Reasoning with Programmatic Verifiers* — Toh et al. (2024).
  [arXiv:2410.12608](https://arxiv.org/abs/2410.12608)
- *GSM8K: Training Verifiers to Solve Math Word Problems* — Cobbe,
  Kosaraju, Bavarian, et al. (2021).
  [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- *HumanEval+ / EvalPlus: Are Your Code Generation Benchmarks
  Reliable?* — Liu, Xia, Wang, Zhang (2023).
  [arXiv:2305.01210](https://arxiv.org/abs/2305.01210)
- *Qwen 2.5 Technical Report* — Qwen Team, Alibaba (2024).
  [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
