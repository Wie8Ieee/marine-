# Checkpoint Recovery and Resume Decision Report

No training or inference was run during this audit.

## Canonical identity required for future runs

- Trash-ICRA19 dataset SHA-256: `5e0f560955eaf8ae4c517aa7eb80215f273dc976e9385f71e87d22f6ffa9e4cf`
- Canonical split manifest SHA-256: `a0d4ad351b536dbfde96926c7500b09c62d2d24bfacc01831c7cf2ed65fa3d94`
- Classes: `plastic`, `bio`, `rov`
- Split seed: 42
- Split: strict sequence-level, frame-balanced 70/15/15

## Decisions

| Run | Checkpoint evidence | Decision | Reason |
|---|---|---|---|
| YOLO-S42-RUNPOD-SEQUENCE | Log and 10+100 epoch histories; no best/last checkpoint in repository, bundle, or upload archives | Retrain from clean initialization for canonical comparison | Cannot evaluate or resume without the original checkpoint; copied result artifacts are insufficient |
| YOLO-S123-LEGACY stage 1 | 22,501,930-byte checkpoint loads successfully as an Ultralytics detection model | Evaluate only as a legacy result | Wrong seed/augmentation/split provenance for the canonical experiment |
| YOLO-S123-LEGACY stage 2 | `best.pt` is zero bytes | Reject | Corrupted/empty artifact |
| FRCNN-S123-PARTIAL | `last.pt` loads at epoch 87 and contains model weights | Evaluate only as legacy; do not resume as an exact continuation | Optimizer, scheduler, AMP scaler, stage, and stage-epoch states are absent; canonical fingerprints are absent |
| FRCNN-S42-STRATIFIED-LEGACY | Best and last checkpoints load; last is epoch 110 | Evaluate only as legacy | Image-level stratified split is noncanonical and checkpoint selection followed legacy behavior |
| SSD-S42-SEQUENCE-LEGACY | Best loads at epoch 39; last loads at epoch 110 with optimizer/scheduler/scaler state | Preserve; evaluate only as legacy | Its sequence audit and label totals do not match the selected canonical manifest |
| SSD-S123-LEGACY | Best/last load at epochs 54/110 | Evaluate only as legacy | Legacy split and color augmentation; last checkpoint lacks optimizer/scheduler/scaler state |

## Archive search

- `marine_runpod.bundle` is structurally valid, but contains the same available checkpoints, including the zero-byte YOLO stage-2 file. It does not recover YOLO seed 42.
- `tmp/sequence_safe_upload_full.tar` reports truncated input and is not a valid recovery source.
- `tmp/sequence_safe_upload.tar.gz` contains no matching checkpoint/history/result paths.

## Resume safety change

TorchVision resume now refuses a checkpoint unless all of the following are present and compatible:

1. model, optimizer, scheduler, AMP scaler, epoch, stage, and stage-epoch state;
2. exact model architecture;
3. exact dataset SHA-256;
4. exact split-manifest SHA-256.

Legacy weights remain usable for explicitly labeled legacy evaluation, but cannot silently become a canonical resumed run.

## Minimum future training required

The minimum publication-grade set is **three clean seed-42 runs**, one for each architecture, against the canonical Trash manifest:

1. YOLOv8s — clean initialization from the documented pretrained base.
2. Faster R-CNN ResNet-50-FPN — clean initialization from the documented pretrained base.
3. SSDLite320 MobileNetV3-Large — clean initialization from the documented pretrained base.

Existing checkpoints cannot reduce this count because none proves an exact match to the canonical dataset and split fingerprints. Two additional seeds per model are optional later for valid mean/SD/CI reporting, but must not be mixed with the heterogeneous legacy runs.

Explicit approval is recorded. River now excludes all 215 members of the 89 conflicting
exact-image groups, leaving an official evaluation subset of 2,185 images and 3,202
objects. Final detection NMS is class-agnostic and occurs once inside each framework.
The immutable seed-42 config passes all data and policy gates. Training remains pending
only because the current machine is CPU-only; canonical execution requires a CUDA GPU.
