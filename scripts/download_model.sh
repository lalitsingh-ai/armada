#!/usr/bin/env bash
# Download a small, tool-calling GGUF model into models/model.gguf.
# Q4_0 is preferred on Arm: llama.cpp repacks Q4_0 weights onto the i8mm/dotprod paths at load.
#
# Override the model with env vars, e.g.:
#   MODEL_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF MODEL_FILE=qwen2.5-3b-instruct-q4_0.gguf ./scripts/download_model.sh
# For gated repos, export HF_TOKEN first.
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-3B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-3b-instruct-q4_0.gguf}"
OUT="${OUT:-models/model.gguf}"

mkdir -p "$(dirname "$OUT")"
URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}?download=true"

echo ">> downloading ${MODEL_REPO}/${MODEL_FILE}"
AUTH=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

curl -L --fail "${AUTH[@]}" -o "$OUT" "$URL"
echo ">> saved $OUT"
