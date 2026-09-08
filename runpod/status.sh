#!/usr/bin/env bash
set -euo pipefail
root="${1:-/workspace/persistent/marine_control}"
find "$root" -maxdepth 3 -type f -name '*.json' -print -exec python -m json.tool {} \;
