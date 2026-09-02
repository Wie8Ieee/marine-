#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BASE_COMMIT="70d022554d77936c592a5f9b980d9d360afc8638"
REPO_ROOT="${MARINE_REPO_ROOT:-/workspace/marine-}"
PERSISTENT_ROOT="/workspace/persistent"

cd "$REPO_ROOT"
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" || {
  echo "HEAD does not match origin/main" >&2
  exit 1
}
git merge-base --is-ancestor "$EXPECTED_BASE_COMMIT" HEAD || {
  echo "Required audited base commit $EXPECTED_BASE_COMMIT is not in HEAD" >&2
  exit 1
}

test "${MARINE_PERSISTENCE_CONFIRMED:-}" = "YES" || {
  echo "Set MARINE_PERSISTENCE_CONFIRMED=YES only after confirming $PERSISTENT_ROOT is persistent storage." >&2
  exit 1
}
mkdir -p "$PERSISTENT_ROOT/marine_environment" "$PERSISTENT_ROOT/marine_smoke" "$PERSISTENT_ROOT/marine_canonical"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
{
  echo "git_commit=$(git rev-parse HEAD)"
  echo "utc_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi
  python - <<'PY'
import platform
import torch
import torchvision
import ultralytics
print("python", platform.python_version())
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY
  python -m pip freeze
  findmnt -T "$PERSISTENT_ROOT" || true
  df -h "$PERSISTENT_ROOT"
} | tee "$PERSISTENT_ROOT/marine_environment/seed42_environment.txt"

python tools/verify_dataset.py \
  --manifest manifests/trash_icra19/canonical_split_manifest.csv \
  --data-root /workspace/datasets/trash-icra19

for config in \
  config_runpod_smoke.yaml \
  config_runpod_yolo_seed42.yaml \
  config_runpod_frcnn_seed42.yaml \
  config_runpod_ssd_seed42.yaml
do
  python marine_3model_experiment.py --config "$config" --preflight-only
done

python marine_3model_experiment.py --config config_runpod_smoke.yaml \
  2>&1 | tee "$PERSISTENT_ROOT/marine_smoke/seed42_pipeline_check.log"
printf '%s\n' "Excluded from official comparison: quick_debug=true; no Test or River evaluation." \
  > "$PERSISTENT_ROOT/marine_smoke/seed42_pipeline_check/EXCLUDED_FROM_OFFICIAL_COMPARISON.txt"

for model in yolo frcnn ssd
do
  config="config_runpod_${model}_seed42.yaml"
  log="$PERSISTENT_ROOT/marine_canonical/${model}_seed42_training.log"
  python marine_3model_experiment.py --config "$config" 2>&1 | tee "$log"
done
