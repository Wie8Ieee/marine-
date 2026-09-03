# Historical artifacts excluded from final comparison

THESE ARTIFACTS ARE RETAINED FOR HISTORY ONLY.
THEY MUST NOT BE USED IN THE FINAL CANONICAL COMPARISON.

| Run/Artifact | Original Path | Model | Seed | Split | Reason Excluded | Evidence |
|---|---|---|---|---|---|---|
| Legacy extracted run | `extracted_marine_3model_comparison/` | mixed | UNCERTAIN | legacy | incomplete provenance | registry/audit |
| Legacy SSD | `.kaggle_ssd_final/` | SSDLite | 123 | legacy | wrong canonical fingerprint | registry |
| Legacy FRCNN | `.kaggle_training_progress_latest/` | Faster R-CNN | 123 | legacy/partial | incomplete run | registry |
| Legacy YOLO | `training_results/` | YOLOv8s | 123 | legacy | corrupted/incompatible artifact | registry |
| Quick runs | `.kaggle_quick_output*` | mixed | UNCERTAIN | debug | failures/debug only | registry |
| Smoke | `tmp/kaggle_smoke_v6_output/` | mixed | 42 | quick-debug | explicit smoke exclusion | marker file |
| Derived figures | `research_figures/` | mixed | UNCERTAIN | mixed | not linked to canonical artifacts | audit |
