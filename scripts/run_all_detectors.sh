#!/usr/bin/env bash
# Run the eight final detector runs: two models, fragment sizes 1 and 5, retention 1 and 5.
# Usage: scripts/run_all_detectors.sh [JOBS_IN_PARALLEL]   (default 2)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
jobs=${1:-2}
for m in reimplemented wide; do for s in 1 5; do for r in 1 5; do echo "$m $s $r"; done; done; done \
  | xargs -P "$jobs" -L 1 "$here/run_detector.sh"
