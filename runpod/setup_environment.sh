#!/usr/bin/env bash
set -euo pipefail
repo="${MARINE_REPO_ROOT:-/workspace/marine_runpod_release}"
cd "$repo"
commit="$(cat RELEASE_COMMIT.txt 2>/dev/null || git rev-parse HEAD)"
test "$(git rev-parse HEAD 2>/dev/null || printf '%s' "$commit")" = "$commit"
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
python -m pip install -r requirements-runpod-cu128.lock
mkdir -p /workspace/persistent/marine_environment /workspace/persistent/marine_control
{
  echo "source_commit=$commit"; echo "interpreter=$(command -v python)"; date -u +"utc=%Y-%m-%dT%H:%M:%SZ"; nvidia-smi
  python - <<'PY'
import platform,torch,torchvision,ultralytics
print("python",platform.python_version()); print("torch",torch.__version__); print("torchvision",torchvision.__version__)
print("ultralytics",ultralytics.__version__); print("cuda",torch.version.cuda); print("cuda_available",torch.cuda.is_available())
assert torch.__version__.startswith("2.10.0+")
assert torchvision.__version__.startswith("0.25.0+")
assert ultralytics.__version__ == "8.4.143"
PY
  python - <<'PY'
from importlib.metadata import distributions
for distribution in sorted(distributions(), key=lambda item: item.metadata.get("Name", "").lower()):
    print(f'{distribution.metadata.get("Name", "unknown")}=={distribution.version}')
PY
} > /workspace/persistent/marine_environment/release_environment.txt
echo "Environment recorded without credentials."
