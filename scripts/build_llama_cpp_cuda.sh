#!/usr/bin/env bash
# Build llama.cpp with CUDA and record the exact commit — the runtime pin
# that benchresult/v1 requires for verified rows.
#
# Usage:
#   ./scripts/build_llama_cpp_cuda.sh [git-ref]
#
# [git-ref] defaults to master. Whatever ref you choose, the script resolves
# and RECORDS the exact commit sha (printed at the end and written next to
# the binary) — pass that sha to `frontier bench --runtime-commit`. Two
# reproductions of a row must be built from the same recorded commit.
#
# Prereqs (Ubuntu): sudo apt install -y build-essential cmake git
#                   plus the CUDA toolkit (nvcc) matching your driver.
set -euo pipefail

REF="${1:-master}"
SRC_DIR="${LLAMA_CPP_SRC:-$HOME/src/llama.cpp}"
INSTALL_DIR="${LLAMA_CPP_INSTALL:-$HOME/.local}"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found — install the CUDA toolkit first:"
    echo "  https://developer.nvidia.com/cuda-downloads (pick your Ubuntu version)"
    echo "  then re-run. (nvidia-smi alone is the driver, not the toolkit.)"
    exit 1
fi
for tool in cmake git make; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "$tool not found. Ubuntu: sudo apt install -y build-essential cmake git"
        exit 1
    }
done

echo "== Fetching llama.cpp @ ${REF} into ${SRC_DIR}"
if [ ! -d "$SRC_DIR/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$SRC_DIR"
fi
git -C "$SRC_DIR" fetch --tags origin
git -C "$SRC_DIR" checkout --detach "$REF" 2>/dev/null || git -C "$SRC_DIR" checkout --detach "origin/$REF"
COMMIT="$(git -C "$SRC_DIR" rev-parse HEAD)"

echo "== Building with CUDA (commit ${COMMIT}, ${JOBS} jobs)"
cmake -S "$SRC_DIR" -B "$SRC_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC_DIR/build" --target llama-server llama-cli -j "$JOBS"

mkdir -p "$INSTALL_DIR/bin"
cp "$SRC_DIR/build/bin/llama-server" "$SRC_DIR/build/bin/llama-cli" "$INSTALL_DIR/bin/"
echo "$COMMIT" > "$INSTALL_DIR/bin/llama_cpp_commit.txt"

echo
echo "== Done"
echo "binary:         $INSTALL_DIR/bin/llama-server"
echo "runtime commit: $COMMIT   (also in $INSTALL_DIR/bin/llama_cpp_commit.txt)"
echo
echo "Record the commit on every bench run:"
echo "  frontier bench --plan <plan.yaml> --suite coding_agent --runtime-commit $COMMIT"
case ":$PATH:" in
    *":$INSTALL_DIR/bin:"*) ;;
    *) echo
       echo "NOTE: $INSTALL_DIR/bin is not on PATH. Add it:"
       echo "  echo 'export PATH=\"$INSTALL_DIR/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc" ;;
esac
