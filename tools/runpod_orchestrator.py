"""Sequential one-GPU orchestration for YOLO, Faster R-CNN, then SSDLite."""
from __future__ import annotations
import argparse, datetime as dt, json, os, subprocess, sys, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runpod_model_runner import reserve  # noqa:E402
from tools.runpod_release import DATASET_SHA256,SPLIT_SHA256,MODEL_CONFIGS,atomic_json,load_and_validate_config,training_config_sha256,source_commit  # noqa:E402

ORDER=("yolo","frcnn","ssd")
def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()

def validate_preflight(path: Path, repo: Path, commit: str) -> dict:
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("status")!="PASS" or data.get("source_commit")!=commit: raise RuntimeError("Stale or failed GPU preflight evidence")
    if data.get("dataset_sha256")!=DATASET_SHA256 or data.get("split_sha256")!=SPLIT_SHA256: raise RuntimeError("Preflight fingerprint mismatch")
    for model in ORDER:
        item=data.get("models",{}).get(model,{})
        cfg=load_and_validate_config(repo/MODEL_CONFIGS[model],model)
        if item.get("status")!="VERIFIED_COMPLETE" or item.get("canonical_training_config_sha256")!=training_config_sha256(cfg):
            raise RuntimeError(f"Missing or stale {model} preflight evidence")
    return data

def run_group(repo: Path,commit: str,group_id: str,configs: Path,environment: Path,control: Path,preflight: bool,evidence: Path|None=None,runner: Path|None=None) -> dict:
    head=source_commit(repo)
    if head!=commit: raise RuntimeError(f"Repository pin mismatch: {head} != {commit}")
    if not preflight:
        if evidence is None: raise RuntimeError("Official training requires --preflight-evidence")
        validate_preflight(evidence,repo,commit)
    state_path=control/"groups"/f"{group_id}.json"
    state={"schema_version":1,"group_id":group_id,"source_commit":commit,"mode":"preflight" if preflight else "official","status":"RUNNING","started_utc":utc(),"models":{m:"PENDING" for m in ORDER}}
    reserve(state_path,state)
    try:
        for model in ORDER:
            state["models"][model]="RUNNING"; atomic_json(state_path,state)
            runner_path=runner or (repo/"tools/runpod_model_runner.py")
            cmd=[sys.executable,"-u",str(runner_path),"--model",model,"--config",str(configs/f"{model}.yaml"),"--repo",str(repo),"--commit",commit,"--mode","preflight" if preflight else "official","--environment",str(environment),"--control",str(control)]
            result=subprocess.run(cmd,cwd=repo)
            if result.returncode:
                state.update({"status":"FAILED","failed_model":model,"return_code":result.returncode,"ended_utc":utc()}); state["models"][model]="FAILED"; atomic_json(state_path,state); return state
            state["models"][model]="VERIFIED_COMPLETE"; atomic_json(state_path,state)
        state.update({"status":"VERIFIED_COMPLETE","return_code":0,"ended_utc":utc()}); atomic_json(state_path,state); return state
    except Exception as exc:
        state.update({"status":"FAILED","exception_type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc(),"ended_utc":utc()}); atomic_json(state_path,state); raise

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,default=Path.cwd()); p.add_argument("--commit",required=True); p.add_argument("--group-id",required=True); p.add_argument("--configs",type=Path,required=True); p.add_argument("--environment",type=Path,required=True); p.add_argument("--control",type=Path,default=Path("/workspace/persistent/marine_control")); p.add_argument("--preflight",action="store_true"); p.add_argument("--preflight-evidence",type=Path); p.add_argument("--runner",type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(); result=run_group(a.repo.resolve(),a.commit,a.group_id,a.configs.resolve(),a.environment.resolve(),a.control.resolve(),a.preflight,a.preflight_evidence.resolve() if a.preflight_evidence else None,a.runner.resolve() if a.runner else None)
    print(json.dumps(result,indent=2)); return 0 if result["status"]=="VERIFIED_COMPLETE" else 1
if __name__=="__main__": raise SystemExit(main())
