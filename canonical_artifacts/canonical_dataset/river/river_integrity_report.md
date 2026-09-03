# River Floating Trash Integrity Report

## Canonical source

- Archive: `.data_inspection/river-floating-trash-datasets.zip`
- Archive SHA-256: `ca5d8e3331d39d79d8f6e6a23ecc63252160b29a6a34c15f533865fd573f7509`
- Dataset SHA-256: `5a8f7762c09d88e9fb823a03cf880e20a707b7c1f0e8253c567a98c086563ce1`
- Images/labels/objects: 2400/2400/3510
- Original split counts: {'test': 481, 'train': 1680, 'val': 239}
- Empty labels: 0
- Trash-ICRA19 exact-image overlap: 0
- Unique image contents: 2274
- Duplicate image groups: 89
- Duplicate extra files: 126
- Duplicate groups with different label files: 89
- Malformed/out-of-range boxes: 0
- Official evaluation subset: 2185 images/3202 objects.
- Evaluation-subset SHA-256: `5f6f1a7ca2f8edef49c1b8049b405d9065a4cb55c7118e690e60a4d6c17a8af1`
- Excluded conflicting images: 215 (all members of conflicting groups).
- Adoption status: canonical full source; official evaluation uses the filtered manifest.

## Evaluation contract

1. Validate the complete 2,400-image/3,510-object source, then evaluate only the conflict-free filtered manifest; never train or select checkpoints on River.
2. Run the three-class Trash-ICRA19 detector unchanged.
3. Collapse every predicted and ground-truth class to class 0 (`trash`).
4. Apply class-agnostic NMS exactly once inside each detector framework (YOLO `agnostic_nms`; patched final detection NMS for TorchVision).
5. Do not apply a second common NMS pass after framework prediction.
6. Match at the documented confidence and IoU thresholds.

## Legacy copies

- The 64-image prepared copy is an invalid partial/debug extraction.
- The 1,416-image prepared copy is incomplete and must not produce official results.
