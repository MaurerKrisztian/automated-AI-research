#!/usr/bin/env bash
# Study 0004's grid: the pilot, the final runs at each model's chosen configuration, and the replication. Every run saves
# its own JSON and is skipped if it already has one, so rerunning a line resumes it. Set SUFFIX=_rerun to write new
# grids and logs instead of skipping the recorded runs; logs/ and results/ are immutable.
# Usage: [SUFFIX=_rerun] ./run_grid.sh pilot | final <task>... | replicate
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
export TRITON_CACHE_AUTOTUNING=1
S="${SUFFIX:-}"
case "${1:-}" in
  pilot)
    for task in mqarmix volatilemix selcopy; do
      $PY -u train.py --grid "pilot2$S" --tasks "$task" --lrs 1e-3 2e-3 4e-3 --seeds 100 --steps-scale 0.5 --extra-evals --no-numerics 2>&1 | grep -v -E "Warning|warn\(" | tee -a "logs/pilot2${S}_$task.log"
      $PY -u train.py --grid "pilot2$S" --tasks "$task" --lrs 2e-3 --seeds 100 --alt --steps-scale 0.5 --extra-evals --no-numerics 2>&1 | grep -v -E "Warning|warn\(" | tee -a "logs/pilot2${S}_$task.log"
    done
    ;;
  final)
    # results/best_config.json maps model/task -> lr, alt (aggregate.py --write-best).
    shift
    for task in "$@"; do
      $PY -u train.py --grid "final$S" --tasks "$task" --best-from results/best_config.json --seeds 0 1 2 --extra-evals 2>&1 | grep -v -E "Warning|warn\(" | tee -a "logs/final${S}_$task.log"
    done
    ;;
  replicate)
    # The rule's one flagged cell, on fresh seeds, as NOTEBOOK.md set out before training.
    $PY -u train.py --grid "replicate$S" --tasks selcopy --models kalman_free_channel gdn mesanet --best-from results/best_config.json --seeds 10 11 12 2>&1 | grep -v -E "Warning|warn\\(" | tee "logs/replicate${S}_selcopy.log"
    ;;
  *) echo "usage: $0 pilot | final <task>... | replicate"; exit 2 ;;
esac
