#!/usr/bin/env bash
# Creates a local virtual environment (.venv next to this script) with a CUDA 12.8
# build of PyTorch, which supports RTX 50-series (Blackwell) GPUs.
# Leaves the system Python and its ROCm PyTorch untouched. Download is roughly 3 GB.
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python numpy
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
