#!/usr/bin/env bash
set -euo pipefail
repo="${MARINE_REPO_ROOT:-/workspace/marine_runpod_release}"; commit="$(cat "$repo/RELEASE_COMMIT.txt")"
test -n "${MARINE_PREFLIGHT_EVIDENCE:-}" || { echo "MARINE_PREFLIGHT_EVIDENCE is required" >&2; exit 2; }
id="${MARINE_GROUP_ID:-canonical_${commit:0:8}_$(date -u +%Y%m%dT%H%M%SZ)}"; configs="/workspace/persistent/marine_control/configs/$id"
python "$repo/tools/build_runpod_configs.py" --repo "$repo" --destination "$configs" --group-id "$id"
python "$repo/tools/runpod_orchestrator.py" --repo "$repo" --commit "$commit" --group-id "$id" --configs "$configs" \
  --environment /workspace/persistent/marine_environment/release_environment.txt --preflight-evidence "$MARINE_PREFLIGHT_EVIDENCE"
