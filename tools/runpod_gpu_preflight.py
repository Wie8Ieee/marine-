"""GPU, persistent-storage and full dataset-integrity gates."""
from __future__ import annotations
import argparse, csv, json, os, platform, shutil, subprocess, sys
from pathlib import Path
from PIL import Image
import torch
import torchvision
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runpod_release import DATASET_SHA256, SPLIT_SHA256, atomic_json, source_commit  # noqa: E402
from tools.verify_dataset import inspect  # noqa: E402

def torchvision_cuda_ops() -> dict:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    from torchvision.ops import nms, roi_align
    boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device="cuda"); scores=torch.tensor([.9,.8],device="cuda")
    keep=nms(boxes,scores,.5)
    feature=torch.randn(1,2,16,16,device="cuda",requires_grad=True)
    pooled=roi_align(feature,torch.tensor([[0.,1.,1.,12.,12.]],device="cuda"),(3,3)); pooled.sum().backward()
    if keep.numel()<1 or feature.grad is None or not torch.isfinite(feature.grad).all(): raise RuntimeError("TorchVision CUDA ops failed")
    return {"nms":"PASS","roi_align_forward_backward":"PASS"}

def verify_images(manifest:Path,data_root:Path)->int:
    rows=list(csv.DictReader(manifest.open(encoding="utf-8",newline="")))
    for row in rows:
        parts=Path(row["image_path"]).parts; path=data_root/Path(*parts[parts.index("images"):])
        with Image.open(path) as image: image.verify()
    return len(rows)

def persistence_probe(root:Path)->dict:
    if os.environ.get("MARINE_PERSISTENCE_CONFIRMED")!="YES": raise RuntimeError("Persistent volume confirmation missing")
    root.mkdir(parents=True,exist_ok=True)
    mount=json.loads(subprocess.run(["findmnt","-J","-T",str(root)],capture_output=True,text=True,check=True).stdout).get("filesystems",[])
    if not mount or mount[0].get("target") in (None,"/"): raise RuntimeError("Persistent root is not on a dedicated mount")
    expected=os.environ.get("MARINE_PERSISTENT_MOUNT_SOURCE")
    if expected and mount[0].get("source")!=expected: raise RuntimeError("Persistent mount source mismatch")
    free=shutil.disk_usage(root).free
    if free<50*1024**3: raise RuntimeError(f"Insufficient persistent free space: {free}")
    probe=root/".marine_persistence_probe"; payload=os.urandom(64)
    with probe.open("wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    if probe.read_bytes()!=payload: raise RuntimeError("Persistent write/read probe mismatch")
    probe.unlink(); return {"status":"PASS","mount":mount[0],"free_bytes":free}

def run(data_root:Path,manifest:Path,persistent_root:Path,output:Path,repo:Path,commit:str)->dict:
    if not torch.cuda.is_available(): raise RuntimeError("GPU PREFLIGHT NOT RUN: CUDA unavailable")
    head=source_commit(repo)
    if head!=commit: raise RuntimeError(f"Repository pin mismatch: {head} != {commit}")
    fingerprint,_,errors=inspect(manifest,data_root=data_root)
    if errors: raise RuntimeError("Dataset integrity errors: "+"; ".join(errors[:20]))
    if fingerprint["dataset_sha256"]!=DATASET_SHA256 or fingerprint["split_manifest_sha256"]!=SPLIT_SHA256: raise RuntimeError("Dataset or split fingerprint mismatch")
    report={"schema_version":1,"status":"PASS","source_commit":commit,"dataset_sha256":DATASET_SHA256,"split_sha256":SPLIT_SHA256,"decoded_images":verify_images(manifest,data_root),"gpu":torch.cuda.get_device_name(0),"gpu_memory_bytes":torch.cuda.get_device_properties(0).total_memory,"torch":torch.__version__,"torchvision":torchvision.__version__,"cuda":torch.version.cuda,"python":platform.python_version(),"torchvision_ops":torchvision_cuda_ops(),"persistence":persistence_probe(persistent_root)}
    atomic_json(output,report); return report

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--persistent-root",type=Path,default=Path("/workspace/persistent")); p.add_argument("--output",type=Path,required=True); p.add_argument("--repo",type=Path,default=Path.cwd()); p.add_argument("--commit",required=True)
    a=p.parse_args(); print(json.dumps(run(a.data_root.resolve(),a.manifest.resolve(),a.persistent_root.resolve(),a.output.resolve(),a.repo.resolve(),a.commit),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
