#!/usr/bin/env bash
set -euo pipefail

MATMUL_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MATMUL_TRITON_DIR="${MATMUL_PROJECT_DIR}/.deps/triton"
MATMUL_TRITON_REV="a5d0d584a77153aeb1e765c02abdc7868a31b751"
MATMUL_PATCH="${MATMUL_PROJECT_DIR}/patches/triton-tlx-warp-specialization.patch"

cd "${MATMUL_PROJECT_DIR}"
uv python install 3.12 --managed-python
MATMUL_PYTHON="$(uv python find 3.12 --managed-python)"

if [[ ! -d "${MATMUL_TRITON_DIR}/.git" ]]; then
    mkdir -p "${MATMUL_PROJECT_DIR}/.deps"
    git clone --filter=blob:none --no-checkout \
        https://github.com/triton-lang/triton.git "${MATMUL_TRITON_DIR}"
    git -C "${MATMUL_TRITON_DIR}" fetch --depth 1 origin "${MATMUL_TRITON_REV}"
    git -C "${MATMUL_TRITON_DIR}" checkout --detach "${MATMUL_TRITON_REV}"
fi

if [[ "$(git -C "${MATMUL_TRITON_DIR}" rev-parse HEAD)" != "${MATMUL_TRITON_REV}" ]]; then
    echo "error: .deps/triton is not at ${MATMUL_TRITON_REV}" >&2
    exit 1
fi

if git -C "${MATMUL_TRITON_DIR}" apply --reverse --check "${MATMUL_PATCH}" 2>/dev/null; then
    : # Patch is already present.
elif git -C "${MATMUL_TRITON_DIR}" apply --check "${MATMUL_PATCH}"; then
    git -C "${MATMUL_TRITON_DIR}" apply "${MATMUL_PATCH}"
else
    echo "error: TLX warp-specialization patch does not apply cleanly" >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]] || \
   [[ "$(readlink -f .venv/bin/python)" != "$(readlink -f "${MATMUL_PYTHON}")" ]]; then
    uv venv --clear --python "${MATMUL_PYTHON}" .venv
fi

MATMUL_GCC_MAJOR="$(c++ -dumpversion | cut -d. -f1)"
MATMUL_MULTIARCH="$(c++ -print-multiarch)"
CPLUS_INCLUDE_PATH="/usr/include/c++/${MATMUL_GCC_MAJOR}:/usr/include/${MATMUL_MULTIARCH}/c++/${MATMUL_GCC_MAJOR}" \
MAX_JOBS="${MAX_JOBS:-64}" \
TRITON_APPEND_CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-Wno-error=attributes -DTRITON_VERSION=3.7.1+gita5d0d584" \
TRITON_EXT_ENABLED=ON \
uv sync --frozen --reinstall-package triton

uv run python -c \
    "from tlx_plugin import load_tlx; load_tlx(); print('TLX compiler plugin: OK')"
