#!/usr/bin/env bash
# GPU benchmark of study 0003 (E3, backlog I-016). From this folder, on an NVIDIA GPU, after scripts/setup_env.sh:
#     bash run_gpu.sh 2>&1 | tee /tmp/run_gpu.out      # in the background; see README, Reproduce, for the times
# It runs the GPU probe, a short smoke test (gpu_bench.py --quick), then the benchmark as five runs, each well under the
# lab's 60-minute limit for one run: parts A-B for N = 16, part B for N = 64, parts C-D, and part B again for each N in
# fresh processes (how stable the times are). Then the tables. Every run writes a fresh log in logs/ (gpu_probe,
# gpu_bench_quick, gpu_bench_n16, gpu_bench_n64, gpu_bench_cd, gpu_bench_rerun_n16, gpu_bench_rerun_n64,
# aggregate_gpu_bench) and every benchmark run a JSON in results/. It never overwrites a log: if a name is taken, the
# next run writes _v2, _v3, ...
# flash-linear-attention's Gated DeltaNet, KDA and MesaNet kernels are timed too when the package is installed
# (fla-core 0.6.0 imports `packaging` without declaring it):
#     uv pip install --python ../../.venv/bin/python fla-core==0.6.0 einops packaging
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

fresh() {  # the first unused name among logs/$1.log, logs/$1_v2.log, logs/$1_v3.log, ...
  local f="logs/$1.log" i=2
  while [ -e "$f" ]; do f="logs/$1_v$i.log"; i=$((i + 1)); done
  echo "$f"
}

bench() {  # bench <log name> <gpu_bench.py arguments...>: run it, and print the results JSON it saved
  local log
  log="$(fresh "$1")"
  shift
  $PY -u gpu_bench.py "$@" 2>&1 | tee "$log" >&2
  grep -o 'results/gpu_bench_[0-9_]*\.json' "$log" | tail -1
}

nvidia-smi
$PY -u ../../scripts/gpu_probe.py 2>&1 | tee "$(fresh gpu_probe)"
$PY -u gpu_bench.py --quick 2>&1 | tee "$(fresh gpu_bench_quick)"
J16="$(bench gpu_bench_n16 --parts A B --N 16)"
J64="$(bench gpu_bench_n64 --parts B --N 64)"
JCD="$(bench gpu_bench_cd --parts C D)"
R16="$(bench gpu_bench_rerun_n16 --parts B --N 16)"
R64="$(bench gpu_bench_rerun_n64 --parts B --N 64)"
python3 aggregate_gpu_bench.py "$J16" "$J64" "$JCD" --rerun "$R16" "$R64" 2>&1 | tee "$(fresh aggregate_gpu_bench)"
echo "done: $J16 $J64 $JCD $R16 $R64"
