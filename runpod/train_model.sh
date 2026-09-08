#!/usr/bin/env bash
set -euo pipefail
model="${1:?usage: train_model.sh yolo|frcnn|ssd}"; case "$model" in yolo|frcnn|ssd);; *) exit 2;; esac
repo="${MARINE_REPO_ROOT:-/workspace/marine_runpod_release}"; commit="$(cat "$repo/RELEASE_COMMIT.txt")"
test -n "${MARINE_PREFLIGHT_EVIDENCE:-}" || { echo "MARINE_PREFLIGHT_EVIDENCE is required" >&2; exit 2; }
python - "$repo" "$commit" "$MARINE_PREFLIGHT_EVIDENCE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]); from tools.runpod_orchestrator import validate_preflight
validate_preflight(Path(sys.argv[3]),Path(sys.argv[1]),sys.argv[2])
PY
id="${MARINE_MODEL_RUN_ID:-canonical_${model}_${commit:0:8}_$(date -u +%Y%m%dT%H%M%SZ)}"; configs="/workspace/persistent/marine_control/configs/$id"
python "$repo/tools/build_runpod_configs.py" --repo "$repo" --destination "$configs" --group-id "$id"
python "$repo/tools/runpod_model_runner.py" --model "$model" --config "$configs/$model.yaml" --repo "$repo" --commit "$commit" --mode official --environment /workspace/persistent/marine_environment/release_environment.txt
