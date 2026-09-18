#!/usr/bin/env bash
# Stop every compute process currently using an NVIDIA GPU on this host.

set -euo pipefail

mapfile -t pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    awk '/^[0-9]+$/ { print }' |
    sort -un
)

if ((${#pids[@]} == 0)); then
  echo "No GPU compute processes found."
  exit 0
fi

printf 'Stopping GPU compute processes: %s\n' "${pids[*]}"
kill -- "${pids[@]}"
