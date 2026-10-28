#!/usr/bin/env bash
# Creates or updates the shared virtual environment .venv at the repository root, with PyTorch and
# requirements.txt, and puts the repository root on its path so `import lab` works from any folder.
#   scripts/setup_env.sh          CUDA 12.8 build of PyTorch, for NVIDIA GPUs including RTX 50-series (~3 GB)
#   scripts/setup_env.sh --cpu    CPU-only PyTorch, for machines without an NVIDIA GPU (~200 MB)
# Uses uv when it is installed (https://docs.astral.sh/uv/), otherwise python3 -m venv and pip.
set -euo pipefail
cd "$(dirname "$0")/.."
INDEX=https://download.pytorch.org/whl/cu128
if [ "${1:-}" = "--cpu" ]; then INDEX=https://download.pytorch.org/whl/cpu; fi
if command -v uv >/dev/null 2>&1; then
  [ -x .venv/bin/python ] || uv venv --python 3.11 .venv
  install() { uv pip install --python .venv/bin/python "$@"; }
else
  [ -x .venv/bin/python ] || python3 -m venv .venv
  install() { .venv/bin/python -m pip install "$@"; }
fi
install "torch==2.11.0" --index-url "$INDEX"
install -r requirements.txt
SITE="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$(pwd)" > "$SITE/airesearch-root.pth"
.venv/bin/python -c "import torch, lab; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
