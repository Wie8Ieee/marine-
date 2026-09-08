# Three-model RunPod deployment (GPU validation pending)

This release prepares, but does not launch, sequential Seed-42 training of YOLOv8s,
Faster R-CNN ResNet-50-FPN, and SSDLite320 MobileNetV3-Large. It deliberately does
not run held-out Test, River, FPS, hyperparameter search, or planned cross-session
resume. Faster smoke Version 11 did **not** establish numerical exact resume; a
single continuous RunPod job avoids depending on that unverified feature but does
not prove numerical reproducibility by itself.

## Canonical configuration audit

| Model | Config | Initialization | Input | Batch | Stages | Freeze policy | Optimizer / schedule | AMP | Workers | Augmentation | Best metric |
|---|---|---|---:|---:|---|---|---|---|---:|---|---|
| YOLOv8s | `config_runpod_yolo_seed42.yaml` | `yolov8s.pt` | 640 | 16 | 10 + 100 | first 10 modules, then all | AdamW 0.01 / cosine | yes | 8 | horizontal flip 0.50; mosaic/MixUp/copy-paste/geometric/HSV off | validation mAP@0.5:0.95 |
| Faster R-CNN ResNet-50-FPN | `config_runpod_frcnn_seed42.yaml` | TorchVision DEFAULT | 640 | 8 | 10 + 100 | backbone, then all | SGD 0.001, momentum 0.9, wd 5e-4 / cosine per stage | yes | 8 | horizontal flip 0.50 only | validation mAP@0.5:0.95 |
| SSDLite320 MobileNetV3-Large | `config_runpod_ssd_seed42.yaml` | TorchVision DEFAULT | 320 | 16 | 10 + 100 | backbone, then all | SGD 0.001, momentum 0.9, wd 5e-4 / cosine per stage | yes | 8 | horizontal flip 0.50 only | validation mAP@0.5:0.95 |

All use Seed 42, classes `plastic,bio,rov`, the canonical sequence-safe split,
dataset SHA-256 `5e0f560955eaf8ae4c517aa7eb80215f273dc976e9385f71e87d22f6ffa9e4cf`,
and split SHA-256 `a0d4ad351b536dbfde96926c7500b09c62d2d24bfacc01831c7cf2ed65fa3d94`.
Model input sizes and native checkpoint formats are intentionally architecture-specific.
The immutable `canonical_split_manifest.csv` is the only source of split membership;
the launcher does not regenerate or rebalance it. Source and materialized membership
must match exactly (missing, extra, moved, and overlapping images/sequences are all zero).
YOLO Stage 2 starts from the non-empty Stage-1 `last.pt`, and the transition records
the source checkpoint SHA-256. TorchVision checkpoints are published atomically.

## Storage and package setup

Attach a persistent volume and mount it so `/workspace/persistent` resolves to a
dedicated mount (not merely the container root filesystem). Reserve at least 50 GiB.
Upload and unzip the package, and place canonical data under
`/workspace/datasets/trash-icra19/{images,labels}`.

```bash
cd /workspace
unzip marine_runpod_three_models.zip
cd marine_runpod_release
export MARINE_REPO_ROOT=/workspace/marine_runpod_release
export MARINE_PERSISTENCE_CONFIRMED=YES
export MARINE_PERSISTENT_MOUNT_SOURCE='<source shown by findmnt>' # optional stronger gate
bash runpod/setup_environment.sh
```

The lock derives from the exercised environment: Python 3.12.13, PyTorch
2.10.0+cu128, TorchVision 0.25.0+cu128, and Ultralytics 8.4.143. Setup records the
actual interpreter and complete installed environment.

## Validate inputs and perform real GPU preflight

```bash
bash runpod/validate_inputs.sh
bash runpod/preflight_all.sh | tee /workspace/persistent/marine_control/preflight_all.log
```

The preflight uses each real training entrypoint and architecture, both stages,
validation, official batch/AMP/workers/preprocessing, fresh-process checkpoint
loading, and actual artifact verifiers. Test-only overrides are 1+1 epochs,
32 samples (YOLO/Faster), 33 samples (SSDLite, exercising its `drop_last=True`
singleton-final-batch policy), and separate outputs. It checks CUDA NMS and ROI
Align backward, decodes every canonical image, validates labels/fingerprints,
records peak memory, and verifies the persistent mount.

The command prints `PREFLIGHT_EVIDENCE=...`. Official commands reject missing,
failed, stale, wrong-commit, wrong-config, or wrong-fingerprint evidence.
Training-only runs have no River dependency and must not create Test, River, FPS,
prediction, or qualitative-analysis artifacts—even empty placeholders. Their artifact
verifier fails closed if any such output exists.

## Train unattended

Only after all GPU preflight artifacts pass:

```bash
export MARINE_PREFLIGHT_EVIDENCE=/workspace/persistent/marine_environment/preflight_evidence_<id>.json
tmux new-session -d -s marine3 \
  "cd /workspace/marine_runpod_release && bash runpod/train_all.sh 2>&1 | tee /workspace/persistent/marine_control/train_all.log"
tmux attach -t marine3
```

Order is YOLO, Faster R-CNN, SSDLite, one process at a time. Each model receives a
fresh run ID/output and `resume:false`; no automatic retry occurs. The next model
starts only after the prior artifact verifier passes. Reinvocation with the same
ID is blocked atomically and never deletes or overwrites outputs.

Single model (also gated by all-model preflight evidence):

```bash
export MARINE_PREFLIGHT_EVIDENCE=/workspace/persistent/marine_environment/preflight_evidence_<id>.json
tmux new-session -d -s marine-one \
  "cd /workspace/marine_runpod_release && bash runpod/train_model.sh yolo"
# Replace yolo with frcnn or ssd.
```

## Monitor, verify, and export

```bash
bash runpod/status.sh
tail -F /workspace/persistent/marine_control/train_all.log
bash runpod/export_all.sh <GROUP_ID>
sha256sum /workspace/persistent/marine_exports/<GROUP_ID>/*.tar.gz
```

Exports require `VERIFIED_COMPLETE` manifests and exclude prepared data,
predictions, qualitative/Test/River/FPS outputs, and per-epoch temporary YOLO
checkpoints. They include final `best`/`last`, histories, configs, environment,
logs/provenance, sizes, and SHA-256 inventories. A continuous run records exact
resume as `NOT_APPLICABLE`, never as a fabricated PASS.

On failure, inspect `/workspace/persistent/marine_control/groups`, `executions`,
and `logs`. Preserve the output and state files; do not delete, resume, or retry
without a separate evidence-based decision.
