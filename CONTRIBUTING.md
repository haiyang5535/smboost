# Contributing to SMBoost Harness

First off, thank you for considering contributing to SMBoost Harness! We are building a production-grade execution harness for small open models (2B–7B), and community contributions shape the roadmap directly.

---

## 1. Quick Local Setup

Fork and clone the repository:

```bash
git clone https://github.com/<your-username>/smboost.git
cd smboost
```

Set up a Python virtual environment (Python 3.10+ required):

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,benchmark]"
```

---

## 2. Models Setup

Because model weights are too large for Git, you must download the specific **Qwen3.5 GGUF** quantizations we use for evaluation directly from Hugging Face into the `models/` directory.

We use the `Q4_K_M` quantizations from the Unsloth Hugging Face repository.

Run the following commands from the project root:

```bash
mkdir -p models
cd models

# 0.8B Model
wget "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf"

# 2B Model
wget "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf"

# 4B Model
wget "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"

# 9B Model
wget "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf"

cd ..
```

---

## 3. Benchmark Dataset Setup

Our LiveCodeBench `Hard` test set (`livecodebench_hard_v1.jsonl`) and other raw data logs are ignored by Git due to their size.

You can download the pre-processed benchmark dataset directly from our Hugging Face dataset repository. Run the following commands:

```bash
# Ensure the Hugging Face CLI is installed
pip install -U "huggingface_hub[cli]"

# Download the dataset directly into the benchmarks directory
hf download hai5535/smboost-lcb-hard livecodebench_hard_v1.jsonl --local-dir benchmarks/data/ --repo-type dataset
```

---

## 4. Running Benchmarks and Tests

To ensure everything runs correctly:

**Run the unit tests:**

```bash
pytest tests/ -q
```

**Run a functional probe (with `qwen3.5:2b` loaded locally):**
_Ensure your LLaMA server is running on port 8000 via `llama_cpp.server`._

```bash
SMBOOST_LLM_BACKEND=server \
SMBOOST_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
python3 scripts/smoke_pipeline.py qwen3.5:2b
```

---

## 5. Development Workflow & Pull Requests

1. **Create a branch**: `git checkout -b fix/your-feature-name`
2. **Commit your changes**: Write clear, descriptive commit messages.
3. **Ensure tests pass**: Run `pytest tests/` before opening a PR.
4. **Push and Submit**: Push to your fork and submit a Pull Request against our `main` branch.

_Early contributors (first 20 PRs merged) get co-authorship credit on the public benchmark paper releasing alongside the v1.0 milestone._
