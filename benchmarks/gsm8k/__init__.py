"""GSM8K benchmark package — math word problems with numeric final answers.

Research (Prove, Cobbe et al.) indicates GSM8K is the harness "sweet zone"
for small open-weight models: 0.5B/1B/2B all lift when a program verifier
sits between generator and final answer.

Public entry points:
    - ``loader.load_tasks(n)``: download/cached HuggingFace ``openai/gsm8k`` test split
    - ``prompt.build_prompt(question)``: CoT-inducing prompt format
    - ``scorer.extract_answer(completion)`` and ``scorer.score(completion, expected)``
    - ``runner.run_baseline(tasks, model, ...)``: raw-mode runner, mirrors humaneval_plus
"""
