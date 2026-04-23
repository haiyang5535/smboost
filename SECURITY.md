# Security

## Reporting a vulnerability

Please email `w.haiyang@outlook.com` with `[smboost-security]` in the subject.
Do not open a public GitHub issue for suspected vulnerabilities.

## Trust boundary: LiveCodeBench test cases

The LiveCodeBench-Hard benchmark dataset
(`benchmarks/data/livecodebench_hard_v1.jsonl`) is downloaded from a public
Hugging Face dataset (`hai5535/smboost-lcb-hard`) per `CONTRIBUTING.md` §3.
Historically, LCB upstream encodes some test cases as
base64(zlib(pickle(...))). `pickle.loads` on untrusted data is a
remote-code-execution vector: a malicious dataset maintainer (or a compromise
of the HF repo) could ship a payload that runs arbitrary code when the
sandbox deserializes a test case.

### Default behavior

`benchmarks/livecodebench/sandbox.decode_test_cases` refuses to deserialize
pickled entries by default. It silently skips them (no crash) and prints a
one-shot warning to stderr the first time it encounters one in a process.
Native JSON test cases are always accepted.

This means a fresh checkout can run the sandbox test suite and the benchmark
runner without ever invoking `pickle.loads`. Tasks whose ground truth was
pickle-only will simply be treated as "no test cases found" and scored as
failed, which is safer than silently executing untrusted code.

### Opt-in: operator accepts the risk

If you trust the dataset source (e.g., you built it yourself via a
`scripts/process_lcb.py`-style pipeline, or you have vetted the upstream HF
repo) you can opt back in by setting:

```bash
export SMBOOST_ALLOW_PICKLE_TEST_CASES=1
```

Any other value (`0`, `false`, unset) keeps the safe default.

### Longer-term mitigation

The proper fix is to preprocess the dataset once, at download time, into a
JSON-only format and host that derivative in our own HF repo. That work is
tracked separately; this file documents the interim gate.
