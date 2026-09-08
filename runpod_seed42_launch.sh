#!/usr/bin/env bash
set -euo pipefail
repo="${MARINE_REPO_ROOT:-/workspace/marine_runpod_release}"
exec bash "$repo/runpod/train_all.sh" "$@"
