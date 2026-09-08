#!/usr/bin/env bash
set -euo pipefail
repo="${MARINE_REPO_ROOT:-/workspace/marine_runpod_release}"; commit="$(cat "$repo/RELEASE_COMMIT.txt")"
bash "$repo/runpod/validate_inputs.sh"
id="preflight_${commit:0:8}_$(date -u +%Y%m%dT%H%M%SZ)"; configs="/workspace/persistent/marine_control/configs/$id"
python "$repo/tools/build_runpod_configs.py" --repo "$repo" --destination "$configs" --group-id "$id" --preflight
python "$repo/tools/runpod_orchestrator.py" --repo "$repo" --commit "$commit" --group-id "$id" --configs "$configs" --preflight \
  --environment /workspace/persistent/marine_environment/release_environment.txt
evidence="/workspace/persistent/marine_environment/preflight_evidence_${id}.json"
python "$repo/tools/create_preflight_evidence.py" --repo "$repo" --commit "$commit" --group-id "$id" --output "$evidence" --gpu-evidence /workspace/persistent/marine_environment/gpu_input_preflight.json
echo "PREFLIGHT_EVIDENCE=$evidence"
