# RunPod Marine Debris 3-Model Comparison

This package is written to support the IEEE paper experiment:

**YOLOv8s vs Faster R-CNN ResNet-50-FPN vs MobileNet SSD / SSDLite320 MobileNetV3-Large**

It prepares a clean Trash-ICRA19 split, trains all three models, evaluates the held-out **test** split, and performs **class-agnostic cross-domain** testing on River Floating Trash images.

## Why this version fixes the previous notebook issues

1. Evaluation is done on the **test split**, not the validation split.
2. The dataset size and split counts are exported automatically to `dataset_summary.csv`, so the paper can use real counts instead of guessed numbers.
3. River labels are remapped to one class (`trash`) before cross-domain evaluation. This prevents the common YOLO error where labels with class IDs greater than 0 are ignored when `nc=1`.
4. Cross-domain mAP is computed as object-localization mAP, not accidentally replaced by F1-score.
5. The same split is used for all three architectures.

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

## How to run

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
results_overall_test.csv
results_per_class_map.csv
results_cross_domain.csv
results_for_paper.md
fig_speed_accuracy.png
used_config.yaml
```

Use `results_for_paper.md` and the CSV files to update the IEEE paper tables.

## Important wording for the paper

Use this wording only if the full run completes successfully:

> All three models were evaluated on the same held-out Trash-ICRA19 test split. Cross-domain performance was evaluated on the River Floating Trash dataset without retraining using class-agnostic IoU matching at IoU ≥ 0.5.

If YOLOv8s uses mosaic/HSV/flip while Faster R-CNN and SSD do not, keep this limitation in the paper:

> Augmentation parity was not fully controlled because YOLOv8s used additional built-in augmentations. Therefore, part of the observed YOLOv8s advantage may be attributed to augmentation in addition to architecture.

## Expected result behavior

Do not force the code to reproduce old notebook numbers. The correct approach is:

1. Run this script.
2. Use the produced CSV files.
3. Update the paper tables to match the real output.

For a strong and believable paper, the best result is not necessarily the highest possible number; it is the most reproducible and honestly reported number.
