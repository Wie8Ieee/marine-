# RunPod Marine Debris 3-Model Comparison

This package is written to support the IEEE paper experiment:

**YOLOv8s vs Faster R-CNN ResNet-50-FPN vs MobileNet SSD / SSDLite320 MobileNetV3-Large**

It prepares a clean Trash-ICRA19 split, trains all three models, evaluates the held-out **test** split, and performs **class-agnostic cross-domain** testing on River Floating Trash images.

## Why this version fixes the previous notebook issues

1. Evaluation is done on the **test split**, not the validation split.
2. The dataset size and split counts are exported automatically to `dataset_summary.csv`, so the paper can use real counts instead of guessed numbers.
3. River labels are remapped to one class (`trash`) before cross-domain evaluation. This prevents the common YOLO error where labels with class IDs greater than 0 are ignored when `nc=1`.
4. Cross-domain mAP is computed as object-localization mAP, not accidentally replaced by F1-score.
5. The same leakage-controlled split and confidence threshold are used for all three architectures.
6. Exact duplicates are removed before splitting. Sequence splitting is strict by default, refuses an image-level fallback, and exports `sequence_split_audit.csv`.
7. The first canonical pass uses seed 42 only. Additional homogeneous seeds may be added later; mean/SD/CI are never reported for `n=1`.
8. The controlled configuration uses horizontal flip only. Color augmentation is disabled because PIL and Ultralytics implement brightness/saturation differently; YOLO-only Mosaic/MixUp/geometric augmentation is also disabled.
9. Hard FP/FN examples are tagged with reproducible small-object, overlap/occlusion, illumination, and low-contrast/turbidity proxy indicators.
10. Training time, checkpoint size, parameter count, peak GPU memory, FPS, and best-effort GFLOPs per frame are exported automatically.
11. Performance is broken down by small/medium/large objects and by reproducible illumination/contrast conditions.
12. Interpretable image features quantify the Trash-to-River domain shift in luminance, contrast, saturation, and edge strength.
13. FPS uses one batch-1 protocol for every model: pre-decoded/resized CPU tensors, 20 warm-up images, up to 120 timed images, CPU-to-GPU transfer, detector forward pass, post-processing/NMS, and CUDA synchronization.
14. TorchVision saves model, optimizer, scheduler, and AMP-scaler state; YOLO resumes either training stage from its `last.pt`. Rerunning the same seed directory continues incomplete epochs when `training.resume: true`.
15. Object-size mAP uses standard COCO area bins after scaling detections from every architecture to a canonical 640×640 canvas; the full measurement contract is exported to `experiment_protocol.json`.
16. TorchVision best-checkpoint selection uses validation mAP@0.5:0.95, and YOLO early stopping is disabled so every architecture receives the configured epoch budget.
17. `run.evaluate: false` now performs training/checkpointing only; test, cross-domain, figures, and result rows are generated when it is `true`.
18. River uses one final class-agnostic NMS inside each framework: Ultralytics `agnostic_nms` for YOLO and a scoped replacement of TorchVision's final class-aware detection NMS. No second NMS pass is applied.
19. `model_size_mb` is the state-dictionary tensor size for every architecture; the format-dependent file size is retained separately as `checkpoint_size_mb`.
20. The complete River source remains fingerprinted at 2,400 images/3,510 objects. Official evaluation excludes all 215 members of 89 conflicting exact-image groups and uses the documented 2,185-image/3,202-object manifest.

Validate the sequence regex without training:

```bash
python marine_3model_experiment.py --config config.yaml --check-sequences-only
```

The command succeeds only when every Trash image matches and writes `sequence_regex_audit.csv` with the inferred sequence for every file.

YOLO saves one checkpoint per epoch so the script can select the exact epoch with the highest validation mAP@0.5:0.95. Allow roughly 2–3 GB of additional temporary storage for a full two-stage YOLOv8s run; these epoch files can be removed after the selected checkpoint and result folder are archived.

## Recommended RunPod setup

Use a RunPod PyTorch GPU template with CUDA, for example an A40 48GB or RTX A6000 48GB pod. The code also runs on T4/A4000, but Faster R-CNN training will be slower.

Open a terminal inside RunPod:

```bash
cd /workspace
unzip runpod_marine_3model.zip
cd runpod_marine_3model
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Do not blindly reinstall `torch` and `torchvision` if the RunPod image already has a working CUDA PyTorch installation. A mismatch between `torch` and `torchvision` can break detection ops such as NMS.

## Dataset layout expected

The script expects YOLO TXT labels.

Typical layout:

```text
/workspace/datasets/trash-icra19/
  images/
    ...jpg/png
  labels/
    ...txt
  data.yaml   optional

/workspace/datasets/river-floating-trash/
  images/
    ...jpg/png
  labels/
    ...txt
```

Each YOLO label row should be:

```text
class_id x_center y_center width height
```

with normalized coordinates.

For Trash-ICRA19, the default class order is:

```text
0 plastic
1 bio
2 rov
```

For the River dataset, all label classes are automatically remapped to one class named `trash` for class-agnostic evaluation.
Exact-image groups with conflicting labels are excluded in full; the audit is written to `river_excluded_conflicting_duplicates.csv`.

## How to run

### Official clean seed-42 launch

The official training phase uses four immutable configs:

- `config_runpod_smoke.yaml` — short pipeline check, explicitly excluded from comparison.
- `config_runpod_yolo_seed42.yaml`
- `config_runpod_frcnn_seed42.yaml`
- `config_runpod_ssd_seed42.yaml`

All official configs set `resume: false`, `evaluate: false`, and `experiment.seeds: [42]`.
They write to separate directories under `/workspace/persistent`. Mount persistent storage
there and upload the relocated canonical Trash dataset to
`/workspace/datasets/trash-icra19` with `images/` and `labels/` beneath it.

After cloning `main`, run:

```bash
export MARINE_PERSISTENCE_CONFIRMED=YES
bash runpod_seed42_launch.sh
```

The launcher verifies that local `HEAD` equals `origin/main`, that audited commit
`70d02255` is an ancestor, records GPU/CUDA/PyTorch/TorchVision/Ultralytics and the full
Python environment, verifies the relocated dataset fingerprint, runs the smoke check
without Test/River evaluation, and then starts the three clean trainings sequentially.
It stops on the first failed prerequisite or training command.

Copy `config_example.yaml` to `config.yaml`:

```bash
cp config_example.yaml config.yaml
nano config.yaml
```

Edit these paths:

```yaml
trash_root: /workspace/datasets/trash-icra19
river_root: /workspace/datasets/river-floating-trash
out_dir: /workspace/runs/marine_3model_comparison
```

Verify hashes, policy gates, and CUDA availability without starting training:

```bash
python marine_3model_experiment.py --config config.yaml --preflight-only
```

Do not change the provenance hashes or approval policy values. The preflight must pass before the full command is used.

To combine seed runs downloaded into separate directories without retraining:

```bash
python marine_3model_experiment.py --aggregate-root /workspace/downloaded_runs \
  --out-dir /workspace/aggregated_statistics
```

Run a quick debug first:

```bash
python marine_3model_experiment.py --config config.yaml --quick-debug
```

Then run the full experiment:

```bash
python marine_3model_experiment.py --config config.yaml
```

## Main output files

Inside `out_dir`:

```text
dataset_summary.csv
deduplication_report.csv
results_all_runs.csv
results_mean_sd.csv
used_config.yaml
runs/seed_<N>/results_overall_test.csv
runs/seed_<N>/results_per_class_map.csv
runs/seed_<N>/results_cross_domain.csv
runs/seed_<N>/qualitative_errors/
runs/seed_<N>/results_for_paper.md
```

Use `results_for_paper.md` and the CSV files to update the IEEE paper tables.

## Important wording for the paper

Use this wording only if the full run completes successfully:

> All three models were evaluated on the same held-out Trash-ICRA19 test split. Cross-domain performance was evaluated on the River Floating Trash dataset without retraining using class-agnostic IoU matching at IoU ≥ 0.5.

Methodology wording for the controlled comparison:

> Augmentation and the Precision/Recall/F1 confidence operating point were controlled across all three architectures. Dataset partitioning was fixed across training seeds and performed at sequence level where sequence identity was available.

## Expected result behavior

Do not force the code to reproduce old notebook numbers. The correct approach is:

1. Run this script.
2. Use the produced CSV files.
3. Update the paper tables to match the real output.

For a strong and believable paper, the best result is not necessarily the highest possible number; it is the most reproducible and honestly reported number.
