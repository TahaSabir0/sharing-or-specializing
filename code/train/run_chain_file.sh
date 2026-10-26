#!/bin/bash
# Taha-only wrapper: run a newline-delimited config queue through run_chain.sh.
# Blank lines and comments are ignored. The underlying runner keeps the normal
# GPU-1 guard, environment activation, logging, and continue-after-failure policy.
set -euo pipefail
JOBFILE="${1:?usage: run_chain_file.sh <config-job-file>}"
[ -f "$JOBFILE" ] || { echo "missing job file: $JOBFILE" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
mapfile -t CONFIGS < <(sed 's/[[:space:]]*#.*$//' "$JOBFILE" | sed '/^[[:space:]]*$/d')
[ ${#CONFIGS[@]} -gt 0 ] || { echo "no configs in $JOBFILE" >&2; exit 2; }
exec bash "$HERE/run_chain.sh" "${CONFIGS[@]}"
