#!/usr/bin/env bash
set -euo pipefail
group="${1:?usage: export_all.sh GROUP_ID}"; repo="${MARINE_REPO_ROOT:-/workspace/marine_runpod_release}"
destination="/workspace/persistent/marine_exports/$group"; mkdir -p "$destination"
for model in yolo frcnn ssd; do python "$repo/tools/export_training_artifacts.py" --output "/workspace/persistent/marine_runs/$group/$model" --destination "$destination/${model}_verified_training.tar.gz"; done
