"""Build the evidence-based registry for existing experiment artifacts only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manifests/experiments"


def digest(relative: str | None) -> str:
    if not relative:
        return "Unknown / Not found"
    path = ROOT / relative
    if not path.is_file() or path.stat().st_size == 0:
        return "Unknown / Not found"
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def size(relative: str | None) -> int | str:
    if not relative or not (ROOT / relative).is_file():
        return "Unknown / Not found"
    return (ROOT / relative).stat().st_size


def item(
    experiment_id: str,
    model: str,
    status: str,
    config: str | None = None,
    history: str | None = None,
    best: str | None = None,
    last: str | None = None,
    evaluations: tuple[str, ...] = (),
    **values: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "experiment_id": experiment_id,
        "model_architecture": model,
        "status": status,
        "training_seed": "Unknown / Not found",
        "split_seed": "Unknown / Not found",
        "dataset_fingerprint": "Unknown / Not found",
        "split_manifest_hash": "Unknown / Not found",
        "river_fingerprint": "Unknown / Not found",
        "image_size": "Unknown / Not found",
        "batch_size": "Unknown / Not found",
        "epochs_planned": "Unknown / Not found",
        "epochs_completed": "Unknown / Not found",
        "optimizer": "Unknown / Not found",
        "learning_rate": "Unknown / Not found",
        "scheduler": "Unknown / Not found",
        "augmentation": "Unknown / Not found",
        "initialization": "pretrained COCO weights",
        "hardware": "Unknown / Not found",
        "software_versions": "Unknown / Not found",
        "config_path": config or "Unknown / Not found",
        "config_sha256": digest(config),
        "best_checkpoint_path": best or "Unknown / Not found",
        "best_checkpoint_sha256": digest(best),
        "best_checkpoint_bytes": size(best),
        "last_checkpoint_path": last or "Unknown / Not found",
        "last_checkpoint_sha256": digest(last),
        "last_checkpoint_bytes": size(last),
        "history_path": history or "Unknown / Not found",
        "history_sha256": digest(history),
        "evaluation_paths": ";".join(evaluations) if evaluations else "Unknown / Not found",
        "evaluation_sha256": ";".join(digest(path) for path in evaluations) if evaluations else "Unknown / Not found",
        "thresholds": "Unknown / Not found",
        "evaluator_version": "Unknown / Not found",
        "eligible_for_final_comparison": False,
        "eligibility_reason": "No proof of canonical dataset/split fingerprint linkage",
    }
    result.update(values)
    return result


def build() -> list[dict[str, object]]:
    standard_thresholds = "confidence=0.25;iou_match=0.50;river_class_agnostic_nms=0.50"
    return [
        item(
            "FRCNN-S42-STRATIFIED-LEGACY", "Faster R-CNN ResNet-50-FPN", "complete-noncanonical",
            config="extracted_marine_3model_comparison/marine_3model_comparison/used_config.yaml",
            history="extracted_marine_3model_comparison/marine_3model_comparison/torchvision/frcnn/history.csv",
            best="extracted_marine_3model_comparison/marine_3model_comparison/torchvision/frcnn/best.pt",
            last="extracted_marine_3model_comparison/marine_3model_comparison/torchvision/frcnn/last.pt",
            evaluations=("extracted_marine_3model_comparison/marine_3model_comparison/results_overall_test.csv", "extracted_marine_3model_comparison/marine_3model_comparison/frcnn_river_cross_domain_manual.csv"),
            training_seed=42, split_seed=42, image_size=640, batch_size=8,
            epochs_planned=110, epochs_completed=110, optimizer="SGD momentum=0.9 weight_decay=5e-4",
            learning_rate=0.001, scheduler="CosineAnnealingLR", augmentation="legacy/insufficiently documented",
            hardware="Unknown / Not found", thresholds="conf_torch=0.50;iou_match=0.50",
            eligibility_reason="Image-level stratified legacy split; best checkpoint selected under legacy criterion",
        ),
        item(
            "YOLO-S123-LEGACY", "YOLOv8s", "corrupted",
            config="training_results/yolov8s_seed_123/marine_3model_comparison/runs/seed_123/used_config.yaml",
            best="training_results/yolov8s_seed_123/marine_3model_comparison/runs/seed_123/yolo/yolov8s_stage1/weights/best.pt",
            last="training_results/yolov8s_seed_123/marine_3model_comparison/runs/seed_123/yolo/yolov8s_stage2/weights/best.pt",
            evaluations=("training_results/yolov8s_seed_123/marine_3model_comparison/runs/seed_123/results_overall_test.csv", "training_results/yolov8s_seed_123/marine_3model_comparison/runs/seed_123/results_cross_domain.csv"),
            training_seed=123, split_seed=42, image_size=640, batch_size=16,
            epochs_planned=110, epochs_completed="Unknown / Not found", optimizer="AdamW", learning_rate=0.01,
            scheduler="cosine", augmentation="flip=0.5;brightness=0.2;saturation=0.2",
            hardware="Tesla T4 (inferred from associated Kaggle family, not run-bound)", thresholds=standard_thresholds,
            eligibility_reason="Stage-2 checkpoint is zero bytes; history absent; noncanonical split",
        ),
        item(
            "SSD-S123-LEGACY", "SSDLite320 MobileNetV3-Large", "complete-noncanonical",
            config=".kaggle_ssd_final/marine_3model_comparison/runs/seed_123/used_config.yaml",
            history=".kaggle_ssd_final/marine_3model_comparison/runs/seed_123/torchvision/ssd/history.csv",
            best=".kaggle_ssd_final/marine_3model_comparison/runs/seed_123/torchvision/ssd/best.pt",
            last=".kaggle_ssd_final/marine_3model_comparison/runs/seed_123/torchvision/ssd/last.pt",
            evaluations=(".kaggle_ssd_final/marine_3model_comparison/runs/seed_123/results_overall_test.csv", ".kaggle_ssd_final/marine_3model_comparison/runs/seed_123/results_cross_domain.csv"),
            training_seed=123, split_seed=42, image_size=320, batch_size=16,
            epochs_planned=110, epochs_completed=110, optimizer="SGD momentum=0.9 weight_decay=5e-4",
            learning_rate=0.001, scheduler="CosineAnnealingLR", augmentation="flip=0.5;brightness=0.2;saturation=0.2",
            hardware="Tesla T4", software_versions="torch=2.10.0+cu128;cuda=12.8;python=3.12.13",
            thresholds=standard_thresholds,
            eligibility_reason="Legacy sequence-count-balanced split; no canonical fingerprint",
        ),
        item(
            "FRCNN-S123-PARTIAL", "Faster R-CNN ResNet-50-FPN", "partial-not-resumable",
            config=".kaggle_training_progress_latest/marine_3model_comparison/runs/seed_123/used_config.yaml",
            history=".kaggle_training_progress_latest/marine_3model_comparison/runs/seed_123/torchvision/frcnn/history.csv",
            best=".kaggle_training_progress_latest/marine_3model_comparison/runs/seed_123/torchvision/frcnn/best.pt",
            last=".kaggle_training_progress_latest/marine_3model_comparison/runs/seed_123/torchvision/frcnn/last.pt",
            training_seed=123, split_seed=42, image_size=640, batch_size=8,
            epochs_planned=110, epochs_completed=87, optimizer="SGD momentum=0.9 weight_decay=5e-4",
            learning_rate=0.001, scheduler="CosineAnnealingLR", augmentation="flip=0.5;brightness=0.2;saturation=0.2",
            hardware="Tesla T4", software_versions="torch=2.10.0+cu128;cuda=12.8;python=3.12.13",
            thresholds=standard_thresholds,
            eligibility_reason="Stopped at epoch 87; checkpoint has no optimizer/scheduler/scaler state; noncanonical split",
        ),
        item(
            "SSD-S42-SEQUENCE-LEGACY", "SSDLite320 MobileNetV3-Large", "complete-noncanonical",
            config="training_results/ssdlite_seed_42_kaggle/marine_3model_comparison/runs/seed_42/used_config.yaml",
            history="training_results/ssdlite_seed_42_kaggle/marine_3model_comparison/runs/seed_42/torchvision/ssd/history.csv",
            best="training_results/ssdlite_seed_42_kaggle/marine_3model_comparison/runs/seed_42/torchvision/ssd/best.pt",
            last="training_results/ssdlite_seed_42_kaggle/marine_3model_comparison/runs/seed_42/torchvision/ssd/last.pt",
            evaluations=("training_results/ssdlite_seed_42_kaggle/marine_3model_comparison/runs/seed_42/results_overall_test.csv", "training_results/ssdlite_seed_42_kaggle/marine_3model_comparison/runs/seed_42/results_cross_domain.csv"),
            training_seed=42, split_seed=42, image_size=320, batch_size=16,
            epochs_planned=110, epochs_completed=110, optimizer="SGD momentum=0.9 weight_decay=5e-4",
            learning_rate=0.001, scheduler="CosineAnnealingLR", augmentation="flip=0.5;brightness=0;saturation=0",
            hardware="Tesla T4", software_versions="torch=2.10.0+cu128;cuda=12.8;python=3.12.13",
            split_manifest_hash=digest("training_results/ssdlite_seed_42_kaggle/marine_3model_comparison/sequence_split_audit.csv"),
            thresholds=standard_thresholds,
            eligibility_reason="Its sequence audit and label/object totals differ from the chosen canonical manifest",
        ),
        item(
            "YOLO-S42-RUNPOD-SEQUENCE", "YOLOv8s", "partial-not-resumable",
            config="tmp/s3_volume_audit/used_config.yaml",
            history="research_figures/source/yolo_stage2.csv",
            evaluations=("research_figures/source/results_overall_test.csv",),
            training_seed=42, split_seed=42, image_size=640, batch_size=16,
            epochs_planned=110, epochs_completed=110, optimizer="AdamW", learning_rate=0.01,
            scheduler="cosine", augmentation="flip=0.5;brightness=0;saturation=0;mosaic=0;mixup=0;copy_paste=0",
            hardware="NVIDIA GeForce RTX 4090", thresholds=standard_thresholds,
            eligibility_reason="Log proves 10+100 epochs, but final best/last checkpoints are absent and copied evaluation provenance is incomplete",
        ),
        item(
            "RESEARCH-FIGURES-DERIVED-S42", "Mixed three-model derived artifacts", "unverified",
            history="research_figures/source/frcnn_history.csv",
            evaluations=("research_figures/source/results_overall_test.csv", "research_figures/source/results_per_class_map.csv"),
            training_seed=42, split_seed=42, epochs_completed="110 rows per model source",
            eligibility_reason="Derived figures/CSVs are not fully linked to original run directories, checkpoints, configs, and manifests",
        ),
        item("QUICK-20260814-01", "None", "failed", config=".kaggle_quick_output/marine_3model_comparison/used_config.yaml", history=".kaggle_quick_output/marine-three-model-controlled-comparison.log", epochs_completed=0, hardware="CPU", eligibility_reason="Failed: Trash dataset path contained no images"),
        item("QUICK-20260814-02", "None", "failed", config=".kaggle_quick_output_v2/marine_3model_comparison/used_config.yaml", history=".kaggle_quick_output_v2/marine-three-model-controlled-comparison.log", epochs_completed=0, hardware="CPU", eligibility_reason="Data scan/debug ended without training evidence"),
        item("QUICK-20260814-03", "None", "failed", config=".kaggle_quick_output_v3/marine_3model_comparison/used_config.yaml", history=".kaggle_quick_output_v3/marine-three-model-controlled-comparison.log", epochs_completed=0, hardware="Tesla T4", eligibility_reason="River validation failed at 64 images/91 objects"),
        item("RUNPOD-FRCNN-S42-FAILED", "Faster R-CNN ResNet-50-FPN", "failed", config="tmp/s3_volume_audit/used_config.yaml", history="tmp/s3_volume_audit/training_sequence_safe.log", training_seed=42, split_seed=42, image_size=640, batch_size=8, epochs_planned=110, epochs_completed=0, hardware="NVIDIA GeForce RTX 4090", eligibility_reason="BatchNorm failure on a singleton training batch; no checkpoint/result"),
    ]


def main() -> None:
    records = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "experiment_registry.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUTPUT / "experiment_registry.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    groups = {
        "Eligible for final comparison": [r for r in records if r["eligible_for_final_comparison"]],
        "Recover/evaluate only": [r for r in records if r["status"] in {"partial-not-resumable", "corrupted"}],
        "Complete but noncanonical": [r for r in records if r["status"] == "complete-noncanonical"],
        "Exclude": [r for r in records if r["status"] in {"failed", "unverified"}],
    }
    lines = ["# Experiment Provenance Audit", "", "No existing run is eligible for the final canonical comparison.", ""]
    for heading, selected in groups.items():
        lines.extend([f"## {heading}", ""])
        if not selected:
            lines.extend(["- None.", ""])
        else:
            lines.extend([f"- `{r['experiment_id']}` — {r['eligibility_reason']}" for r in selected] + [""])
    (OUTPUT / "provenance_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {len(records)} records to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
