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

# Cap build parallelism. An unbounded `-j` makes `make` compile every heavy C++
# translation unit (httplib, llama-adapter, ggml...) at once, each costing ~1-2 GB,
# which exhausts RAM on CI runners and gets the compilers killed (SIGTERM, exit 143:
# "No child processes"). Budget ~2 GB per job and never exceed the core count.
JOBS="${BUILD_JOBS:-}"
if [ -z "$JOBS" ]; then
  CORES="$(nproc)"
  MEM_GB="$(awk '/MemTotal/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
  MEM_JOBS=$(( MEM_GB / 2 ))
  [ "$MEM_JOBS" -lt 1 ] && MEM_JOBS=1
  JOBS="$CORES"
  [ "$MEM_JOBS" -lt "$JOBS" ] && JOBS="$MEM_JOBS"
fi

echo ">> building llama-server with -j$JOBS (cores=$(nproc), mem-aware cap)"
cmake --build "$DIR/build" -j "$JOBS" --target llama-server

echo ">> done: $DIR/build/bin/llama-server"
