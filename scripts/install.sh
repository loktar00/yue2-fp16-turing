#!/usr/bin/env bash
# Create .venv with a CUDA build of torch that includes Turing (sm_75) kernels,
# install the vendored, fp16-patched yue2 package and the web service.
# No flash-attn, no xformers, no vLLM: YuE2 uses PyTorch SDPA, which works on Turing.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install.sh [options]

  --cuda TAG       PyTorch CUDA wheel tag: cu126, cu128 (default), cu129, cu130.
                   cu128 needs NVIDIA driver >= 570; cu126 works with older drivers (>= 560).
  --python BIN     Python interpreter to build the venv with (default: python3; 3.10+, 3.11 tested)
  --venv DIR       venv location (default: ./.venv)
  --no-weights     do not download the model weights afterwards (~7.3 GB)
  --dry-run        print what would be done and exit
  -h, --help       show this help
EOF
}

CUDA_TAG="${YUE2_CUDA:-cu128}"
PYTHON="python3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HERE/.venv"
WEIGHTS=1
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --cuda) CUDA_TAG="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --no-weights) WEIGHTS=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$CUDA_TAG" in cu126|cu128|cu129|cu130) ;; *) echo "unsupported --cuda $CUDA_TAG" >&2; exit 2 ;; esac

run() { echo "+ $*"; if [ "$DRY" = 0 ]; then "$@"; fi; }

echo "repo:   $HERE"
echo "venv:   $VENV"
echo "torch:  2.10.0 ($CUDA_TAG) from https://download.pytorch.org/whl/$CUDA_TAG"
if [ "$DRY" = 0 ]; then
  command -v "$PYTHON" >/dev/null || { echo "python not found: $PYTHON" >&2; exit 1; }
  "$PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2], "need Python 3.10+, have %d.%d" % sys.version_info[:2]'
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv || \
    echo "warning: nvidia-smi not found; is the NVIDIA driver installed?"
fi

run "$PYTHON" -m venv "$VENV"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
run "$PIP" install --upgrade pip setuptools wheel
run "$PIP" install "torch==2.10.0" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
run "$PIP" install -e "$HERE"
run "$PIP" install -r "$HERE/requirements-service.txt"

if [ "$DRY" = 0 ]; then
  "$PY" "$HERE/scripts/check_gpu.py" || true
fi

if [ "$WEIGHTS" = 1 ]; then
  run "$PY" "$HERE/scripts/download_weights.py"
fi

echo
echo "Done. Start the service with: scripts/run.sh   then open http://127.0.0.1:5830/"
