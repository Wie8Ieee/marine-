"""Bind successful per-model preflights to source/config/data/GPU evidence."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runpod_release import DATASET_SHA256,SPLIT_SHA256,MODEL_CONFIGS,atomic_json,load_and_validate_config,training_config_sha256  # noqa:E402

def create(repo:Path,commit:str,group_id:str,output:Path,gpu_evidence:Path)->dict:
    gpu=json.loads(gpu_evidence.read_text(encoding="utf-8"))
    if gpu.get("status")!="PASS" or gpu.get("source_commit")!=commit: raise RuntimeError("GPU/input preflight evidence is stale or failed")
    total_mib=int(gpu["gpu_memory_bytes"])//(1024**2)
    models={}
    for model,name in MODEL_CONFIGS.items():
        canonical=load_and_validate_config(repo/name,model)
        artifact=Path(f"/workspace/persistent/marine_preflight/{group_id}/{model}/training_artifact_manifest.json")
        report=json.loads(artifact.read_text(encoding="utf-8"))
        if report.get("status")!="VERIFIED_COMPLETE" or report.get("execution_mode")!="gpu_preflight" or report.get("source_commit")!=commit: raise RuntimeError(f"{model} preflight artifacts not verified or stale")
        provenance=report.get("provenance",{})
        if provenance.get("dataset_sha256")!=DATASET_SHA256 or provenance.get("split_manifest_sha256")!=SPLIT_SHA256: raise RuntimeError(f"{model} preflight fingerprint mismatch")
        execution_path=Path(f"/workspace/persistent/marine_control/executions/{group_id}_{model}.json")
        execution=json.loads(execution_path.read_text(encoding="utf-8")); peak=execution.get("peak_gpu_memory_mib")
        if peak is None or total_mib-int(peak)<max(2048,total_mib//10): raise RuntimeError(f"{model} GPU memory safety margin is insufficient or unknown")
        models[model]={"status":"VERIFIED_COMPLETE","artifact_manifest":str(artifact),"execution_record":str(execution_path),"peak_gpu_memory_mib":peak,"free_margin_mib":total_mib-int(peak),"canonical_training_config_sha256":training_config_sha256(canonical)}
    result={"schema_version":1,"status":"PASS","source_commit":commit,"group_id":group_id,"dataset_sha256":DATASET_SHA256,"split_sha256":SPLIT_SHA256,"gpu":gpu,"models":models}
    atomic_json(output,result); return result
def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,default=Path.cwd());p.add_argument("--commit",required=True);p.add_argument("--group-id",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--gpu-evidence",type=Path,required=True);a=p.parse_args();create(a.repo.resolve(),a.commit,a.group_id,a.output.resolve(),a.gpu_evidence.resolve());return 0
if __name__=="__main__":raise SystemExit(main())
