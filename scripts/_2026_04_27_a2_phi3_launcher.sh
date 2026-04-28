#!/bin/bash
# A2 launcher for 2026-04-27 improvement pass.
# Switches the llama.cpp server from Qwen 2.5 2B to Phi-3-mini-4k-instruct
# and runs a GSM8K raw + C5 pilot on n=20.
#
# Usage:
#   bash scripts/_2026_04_27_a2_phi3_launcher.sh
#
# After it returns, /tmp/a2_phi3_n20.csv has the result.

set -euo pipefail

# Stop any running llama_cpp.server (we ran 2B for the GSM8K + HE+ pass).
pkill -f "llama_cpp.server" || true
sleep 2

# Boot Phi-3 mini on port 8000.  No --chat_format flag — llama-cpp-python
# falls back to the GGUF's embedded chat template, which Phi-3-mini ships.
nohup python3 -m llama_cpp.server \
    --model models/Phi-3-mini-4k-instruct-q4.gguf \
    --host 127.0.0.1 --port 8000 \
    --n_gpu_layers -1 --n_ctx 4096 \
    > /tmp/llamacpp_phi3.log 2>&1 &

PHI3_PID=$!
echo "phi3 server PID=$PHI3_PID"

# Wait until /v1/models responds.
until curl -fsS http://127.0.0.1:8000/v1/models > /dev/null 2>&1; do sleep 2; done
echo "phi3 server ready"

# Run GSM8K raw + C5 pilot on n=20.
SMBOOST_LLM_BACKEND=server \
SMBOOST_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
SMBOOST_OPENAI_API_KEY=sk-no-key \
SMBOOST_OPENAI_MAX_TOKENS=512 \
python3 -u scripts/run_gates_0_8b.py \
    --stage GSM8K --model phi3:mini-4k \
    --out-csv /tmp/a2_phi3_n20.csv \
    --n 20 --modes raw,C5 2>&1 | tee /tmp/a2_phi3_n20.log
