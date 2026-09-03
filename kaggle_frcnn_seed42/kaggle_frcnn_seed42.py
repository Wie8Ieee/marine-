"""Official canonical Faster R-CNN Seed-42 training on Kaggle GPU."""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
import yaml

REPOSITORY = "https://github.com/Wie8Ieee/marine-.git"; REQUIRED_ANCESTOR = "70d02255"; CONFIG_NAME = "config_runpod_frcnn_seed42.yaml"; MODEL = "frcnn"; WORK_ROOT = Path("/kaggle/working/marine_frcnn_seed42_official")
def run(*args: str) -> None: print("+", " ".join(args), flush=True); subprocess.run(args, check=True)
def data_root() -> Path:
    roots=[p for p in Path("/kaggle/input").rglob("*") if p.is_dir() and (p/"images").is_dir() and (p/"labels").is_dir()]
    if len(roots)!=1: raise RuntimeError(f"Expected one mounted Trash dataset, found {roots}")
    return roots[0]
def main() -> None:
    repo=WORK_ROOT/"repository"; run("git","clone","--branch","main",REPOSITORY,str(repo)); run("git","-C",str(repo),"merge-base","--is-ancestor",REQUIRED_ANCESTOR,"HEAD"); run("git","-C",str(repo),"rev-parse","HEAD"); run(sys.executable,"-m","pip","install","-q","-r",str(repo/"requirements.txt"))
    cfg=yaml.safe_load((repo/CONFIG_NAME).read_text(encoding="utf-8")); cfg["trash_root"]=str(data_root()); cfg["out_dir"]=str(WORK_ROOT/"output"); runtime=WORK_ROOT/"official_runtime_config.yaml"; runtime.parent.mkdir(parents=True,exist_ok=True); runtime.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"); run(sys.executable,str(repo/"marine_3model_experiment.py"),"--config",str(runtime),"--preflight-only"); subprocess.run([sys.executable,str(repo/"marine_3model_experiment.py"),"--config",str(runtime)],check=True,cwd=repo,env=env); run(sys.executable,str(repo/"tools/verify_training_artifacts.py"),"--model",MODEL,"--out-dir",str(WORK_ROOT/"output"),"--config",str(runtime)); (WORK_ROOT/"OFFICIAL_SEED42_RUN.txt").write_text("model=frcnn\nseed=42\nresume=false\nevaluate=false\n",encoding="utf-8")
if __name__ == "__main__": main()
