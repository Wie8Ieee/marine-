# Deep Learning-Based Marine Debris Detection for Smart Water Monitoring: A Comparative Evaluation with Cross-Domain Testing on River Floating Trash Images

This repository is the compact reproducibility record for the verified canonical experiment `canonical_aed8bedd_20260908T234342Z`.

## Dataset and protocol

- Dataset: Trash-ICRA19; 7,684 images and 11,061 annotated objects.
- Classes: `plastic`, `bio`, and `rov`.
- Sequence-safe split: 5,377 train / 1,154 validation / 1,153 test; sequence overlap: 0.
- Seed: 42. Stage 1: 10 frozen epochs. Stage 2: 100 full fine-tuning epochs.
- Best checkpoint: highest validation mAP@0.5:0.95. Test and River were isolated from checkpoint selection.

## Models and verified results

| Model | Best val mAP@0.5:0.95 | Test mAP@0.5 | Test mAP@0.5:0.95 | River mAP@0.5 | FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv8s | 25.939% | 26.7188% | 12.5888% | 10.7686% | 19.0338 |
| Faster R-CNN ResNet50-FPN | 48.264% | 63.4151% | 33.0616% | 28.3911% | 53.9856 |
| SSDLite320 MobileNetV3-Large | 37.523% | 41.4128% | 22.5784% | 11.5868% | 68.6139 |

FPS values represent architecture-specific inference throughput. They are not a compute-normalized comparison.

The model weights and raw datasets are external artifacts and are intentionally excluded from Git. Their SHA-256 identities are recorded under `provenance/`.

Run `python scripts/verify_artifacts.py` to validate this compact release.
