#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python marine_3model_experiment.py --config config.yaml
