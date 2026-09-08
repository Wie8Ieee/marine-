"""Export only verified training artifacts; exclude datasets and evaluations."""
from __future__ import annotations
import argparse, json, os, tarfile, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.runpod_release import atomic_json,sha256_file  # noqa:E402

FORBIDDEN_PARTS={"prepared_datasets","qualitative_errors","predictions","datasets","images","labels"}
FORBIDDEN_NAMES={"results_overall_test.csv","results_cross_domain.csv","fps_results.json"}
def allowed(path:Path,root:Path)->bool:
    rel=path.relative_to(root)
    return not (set(rel.parts)&FORBIDDEN_PARTS) and path.name not in FORBIDDEN_NAMES and not (path.name.startswith("epoch") and path.suffix==".pt")

def export_model(output:Path,destination:Path)->dict:
    manifest=output/"training_artifact_manifest.json"
    if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")).get("status")!="VERIFIED_COMPLETE": raise RuntimeError(f"Unverified model output: {output}")
    files=sorted(path for path in output.rglob("*") if path.is_file() and allowed(path,output))
    destination.parent.mkdir(parents=True,exist_ok=True); temporary=destination.with_name("."+destination.name+".tmp")
    with tarfile.open(temporary,"w:gz") as archive:
        for path in files: archive.add(path,arcname=str(Path(output.name)/path.relative_to(output)),recursive=False)
    os.replace(temporary,destination)
    report={"status":"PASS","source_output":str(output),"archive":str(destination),"archive_sha256":sha256_file(destination),"files":len(files),"excluded_policy":{"parts":sorted(FORBIDDEN_PARTS),"names":sorted(FORBIDDEN_NAMES),"epoch_checkpoints":True}}
    atomic_json(destination.with_suffix(destination.suffix+".manifest.json"),report); return report

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--destination",type=Path,required=True)
    a=p.parse_args(); print(json.dumps(export_model(a.output.resolve(),a.destination.resolve()),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
