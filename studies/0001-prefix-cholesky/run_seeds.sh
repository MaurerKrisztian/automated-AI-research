#!/usr/bin/env bash
# Robustness check for the 64-pair MQAR level: 2 learning rates x 2 extra seeds,
# 3000 steps, d_model 64 / head dim 32 (same setup as the main grid otherwise).
set -euo pipefail
cd "$(dirname "$0")"
for lr in 1e-3 2e-3; do
  for seed in 1 2; do
    ../../.venv/bin/python -u mqar_recall.py --d 64 --heads 2 --pairs 64 --vocab 1024 \
      --lr "$lr" --seed "$seed" --steps 3000 --log-every 1000 \
      --kinds linear deltanet mesa_exact mesa_cg5 --speed-kinds mesa_exact \
      --out "results/seeds/lr${lr}_s${seed}"
  done
done
../../.venv/bin/python aggregate_mqar.py
