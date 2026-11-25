#!/usr/bin/env bash
# Train and test one detector on the paper's setup (pooled datasets, random 75/25 read level split).
# Usage: scripts/run_detector.sh {reimplemented|wide} FRAGMENT_SIZE RETENTION
# Env:   PYTHON (default python3), DEVICE (cpu or mps or cuda, default cpu)
# Output: results/step18_same_as_original/*.txt and *.npz, log in results/logs/step18_same_as_original/
set -euo pipefail
model=${1:?model: reimplemented or wide}; size=${2:?fragment size}; ret=${3:?retention}
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/results/logs/step18_same_as_original"
if [ "$model" = "wide" ]; then export ARCH=wide; suffix="_wide"; else suffix=""; fi
cd "$root/src"
PATIENCE=100 MIN_EPOCHS=500 DEVICE="${DEVICE:-cpu}" "${PYTHON:-python3}" -I ap_step18_same_as_original.py "$size" "$ret" 1000 line 0 10 \
  > "$root/results/logs/step18_same_as_original/run_size${size}_ret${ret}_line_pool10_es100m500${suffix}.log" 2>&1
