"""Create runtime configs using only documented operational overrides."""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runpod_release import MODEL_CONFIGS, load_and_validate_config  # noqa: E402

def build(repo: Path, destination: Path, group_id: str, preflight: bool) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=False)
    result = {}
    for model, name in MODEL_CONFIGS.items():
        source = repo / name
        original = load_and_validate_config(source, model)
        cfg = copy.deepcopy(original)
        root = "/workspace/persistent/marine_preflight" if preflight else "/workspace/persistent/marine_runs"
        cfg["out_dir"] = f"{root}/{group_id}/{model}"
        if preflight:
            cfg["training"]["epochs_head"] = 1
            cfg["training"]["epochs_finetune"] = 1
            cfg["run"]["quick_debug"] = True
            cfg["run"]["preflight_sample_limit"] = 33 if model == "ssd" else 32
        target = destination / f"{model}.yaml"
        target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        load_and_validate_config(target, model, allow_preflight=preflight, allow_output_override=True)
        result[model] = target
    return result

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,default=Path.cwd()); p.add_argument("--destination",type=Path,required=True); p.add_argument("--group-id",required=True); p.add_argument("--preflight",action="store_true")
    a=p.parse_args(); paths=build(a.repo.resolve(),a.destination.resolve(),a.group_id,a.preflight)
    for model,path in paths.items(): print(f"{model}={path}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
