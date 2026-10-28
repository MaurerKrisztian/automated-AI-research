#!/usr/bin/env bash
# Improved exact Mesa (carried inverse + closed-form backward) in the MQAR model:
# training-step speed against the other layers, and recall against the earlier runs.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
$PY -u mqar_recall.py --d 64 --heads 2 --pairs 128 --vocab 1024 --lr 2e-3 --seed 0 --steps 2000 --log-every 1000 \
  --kinds mesa_fast --speed-kinds deltanet linear mesa_exact mesa_fast mesa_cg5 mesa_cg15 mesa_cg30 \
  --out results/fast/n128_s0
for seed in 1 2; do
  $PY -u mqar_recall.py --d 64 --heads 2 --pairs 64 --vocab 1024 --lr 2e-3 --seed "$seed" --steps 3000 \
    --log-every 1000 --kinds mesa_fast --speed-kinds mesa_fast --out "results/fast/n64_s$seed"
done
