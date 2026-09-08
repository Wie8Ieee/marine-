"""Build a deterministic, credential-free ZIP from the exact release commit."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runpod_release import sha256_file  # noqa:E402

PREFIXES=("runpod/","tools/runpod_","tools/verify_dataset.py","tools/verify_training_artifacts.py","tools/build_runpod_configs.py","tools/create_preflight_evidence.py","tools/export_training_artifacts.py","manifests/trash_icra19/")
EXACT={"marine_3model_experiment.py","config_runpod_yolo_seed42.yaml","config_runpod_frcnn_seed42.yaml","config_runpod_ssd_seed42.yaml","requirements-runpod-cu128.lock","README_RUNPOD.md","runpod_seed42_launch.sh"}
def selected(name:str)->bool: return name in EXACT or any(name.startswith(prefix) for prefix in PREFIXES)
def build(repo:Path,destination:Path)->dict:
    if subprocess.run(["git","status","--porcelain"],cwd=repo,capture_output=True,text=True,check=True).stdout.strip(): raise RuntimeError("Package must be built from a clean worktree")
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
    tracked=subprocess.run(["git","ls-files","-z"],cwd=repo,capture_output=True,check=True).stdout.decode().split("\0")
    files=sorted(name for name in tracked if name and selected(name))
    if any("kaggle.json" in name.lower() for name in files): raise RuntimeError("Credential file selected")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in files:
            info=zipfile.ZipInfo("marine_runpod_release/"+name,(2020,1,1,0,0,0)); info.external_attr=0o644<<16; info.compress_type=zipfile.ZIP_DEFLATED
            if name.endswith(".sh"): info.external_attr=0o755<<16
            archive.writestr(info,(repo/name).read_bytes())
        info=zipfile.ZipInfo("marine_runpod_release/RELEASE_COMMIT.txt",(2020,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; archive.writestr(info,commit+"\n")
    return {"commit":commit,"path":str(destination),"sha256":sha256_file(destination),"files":len(files)+1}
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,default=Path.cwd()); p.add_argument("--destination",type=Path,required=True); a=p.parse_args(); print(json.dumps(build(a.repo.resolve(),a.destination.resolve()),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
