#!/usr/bin/env bash
# Build llama.cpp tuned for the host CPU. On Arm64 this enables the i8mm/dotprod matmul paths
# and the KleidiAI micro-kernels; on x86 it builds a native baseline for comparison.
set -euo pipefail

REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp}"
DIR="${LLAMA_CPP_DIR:-llama.cpp}"

if [ ! -d "$DIR" ]; then
  echo ">> cloning $REPO"
  git clone --depth 1 "$REPO" "$DIR"
fi

EXTRA_FLAGS=()
if [ "$(uname -m)" = "aarch64" ]; then
  # Opt into Arm's KleidiAI micro-kernels for quantized matmul.
  EXTRA_FLAGS+=("-DGGML_CPU_KLEIDIAI=ON")
fi

echo ">> configuring (native tuning on)"
cmake -S "$DIR" -B "$DIR/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  "${EXTRA_FLAGS[@]}"

echo ">> building llama-server"
cmake --build "$DIR/build" -j --target llama-server

echo ">> done: $DIR/build/bin/llama-server"
