#!/usr/bin/env bash
set -euo pipefail
repo="${MARINE_REPO_ROOT:-/workspace/marine_runpod_release}"; commit="$(cat "$repo/RELEASE_COMMIT.txt")"
python "$repo/tools/runpod_gpu_preflight.py" --repo "$repo" --commit "$commit" \
  --data-root "${MARINE_DATA_ROOT:-/workspace/datasets/trash-icra19}" \
  --manifest "$repo/manifests/trash_icra19/canonical_split_manifest.csv" \
  --persistent-root /workspace/persistent \
  --output /workspace/persistent/marine_environment/gpu_input_preflight.json
