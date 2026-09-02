# Experiment Provenance Audit

No existing run is eligible for the final canonical comparison.

## Eligible for final comparison

- None.

## Recover/evaluate only

- `YOLO-S123-LEGACY` — Stage-2 checkpoint is zero bytes; history absent; noncanonical split
- `FRCNN-S123-PARTIAL` — Stopped at epoch 87; checkpoint has no optimizer/scheduler/scaler state; noncanonical split
- `YOLO-S42-RUNPOD-SEQUENCE` — Log proves 10+100 epochs, but final best/last checkpoints are absent and copied evaluation provenance is incomplete

## Complete but noncanonical

- `FRCNN-S42-STRATIFIED-LEGACY` — Image-level stratified legacy split; best checkpoint selected under legacy criterion
- `SSD-S123-LEGACY` — Legacy sequence-count-balanced split; no canonical fingerprint
- `SSD-S42-SEQUENCE-LEGACY` — Its sequence audit and label/object totals differ from the chosen canonical manifest

## Exclude

- `RESEARCH-FIGURES-DERIVED-S42` — Derived figures/CSVs are not fully linked to original run directories, checkpoints, configs, and manifests
- `QUICK-20260814-01` — Failed: Trash dataset path contained no images
- `QUICK-20260814-02` — Data scan/debug ended without training evidence
- `QUICK-20260814-03` — River validation failed at 64 images/91 objects
- `RUNPOD-FRCNN-S42-FAILED` — BatchNorm failure on a singleton training batch; no checkpoint/result
