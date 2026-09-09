#!/usr/bin/env python3
import csv, hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASET = "5e0f560955eaf8ae4c517aa7eb80215f273dc976e9385f71e87d22f6ffa9e4cf"
EXPECTED_SPLIT = "a0d4ad351b536dbfde96926c7500b09c62d2d24bfacc01831c7cf2ed65fa3d94"
EXPECTED_WEIGHTS = {
 "yolo": "419181839575c220d457fd3283b9f560466a8ecd7c42c4283cb3ebff73e756e4",
 "frcnn": "aa1dd781efde1279e2cc12567e54640c34d238f9bf2288255c20fa3785db512d",
 "ssd": "eaf4ea46b3483ec861d0695b66f95f8e445907e626a7d9b7a328e472f186338e",
}

def require(ok, message):
 if not ok: raise SystemExit("FAIL: " + message)

release = json.loads((ROOT / "CANONICAL_RELEASE_MANIFEST.json").read_text())
require(release["status"] == "VERIFIED_COMPLETE", "release status")
require(release["dataset_sha256"] == EXPECTED_DATASET, "dataset SHA")
require(release["split_sha256"] == EXPECTED_SPLIT, "split SHA")
models = {m["id"]: m for m in release["models"]}
require(set(models) == set(EXPECTED_WEIGHTS), "model set")

def table(name):
 with (ROOT / "results" / name).open(newline="", encoding="utf-8") as handle:
  return list(csv.DictReader(handle))

test_rows = {row["model"]: row for row in table("test_results.csv")}
river_rows = {row["model"]: row for row in table("river_results.csv")}
fps_rows = {row["model"]: row for row in table("fps_results.csv")}
val_rows = {row["model"]: row for row in table("validation_results.csv")}
per_class_rows = table("per_class_results.csv")
labels = {"yolo": "YOLOv8s", "frcnn": "Faster R-CNN ResNet50-FPN", "ssd": "SSDLite320 MobileNetV3-Large"}
require(len(per_class_rows) == 9, "per-class row count")
for model, expected in EXPECTED_WEIGHTS.items():
 require(models[model]["best_checkpoint_sha256"] == expected, model + " checkpoint SHA record")
 test = models[model]["test"]
 river = models[model]["river"]
 require(model in test_rows and model in river_rows, model + " result table row")
 require(labels[model] in fps_rows and labels[model] in val_rows, model + " summary table row")
 for key in ("map_50", "map_50_95", "precision", "recall", "f1"):
  require(math.isclose(float(test_rows[model][key]), float(test[key]), rel_tol=1e-12), model + " Test table " + key)
  require(math.isclose(float(river_rows[model][key]), float(river[key]), rel_tol=1e-12), model + " River table " + key)
 require(math.isclose(float(fps_rows[labels[model]]["fps"]), float(test["fps"]), rel_tol=1e-12), model + " FPS table")
 require(math.isclose(float(val_rows[labels[model]]["validation_map_50_95"]), float(models[model]["validation_map_50_95"]), rel_tol=1e-12), model + " validation table")
 require(float(test["map_50"]) >= float(test["map_50_95"]), model + " Test mAP ordering")
 require(float(river["map_50"]) >= float(river["map_50_95"]), model + " River mAP ordering")
 for scope in (test, river):
  p, r, f1 = map(float, (scope["precision"], scope["recall"], scope["f1"]))
  expected_f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
  require(math.isclose(f1, expected_f1, rel_tol=1e-10, abs_tol=1e-12), model + " F1 consistency")

for model in EXPECTED_WEIGHTS:
 artifact = ROOT / "manifests" / f"{model}_training_artifact_manifest.json"
 data = json.loads(artifact.read_text())
 require(data["status"] == "VERIFIED_COMPLETE", model + " training manifest status")
 require(data["checkpoints"]["best.pt"]["sha256"] == EXPECTED_WEIGHTS[model], model + " manifest SHA")

hash_text = (ROOT / "provenance/checkpoint_hashes.txt").read_text()
require(all(value in hash_text for value in EXPECTED_WEIGHTS.values()), "checkpoint hash registry")
dataset_text = (ROOT / "provenance/dataset_split_hashes.txt").read_text()
require(EXPECTED_DATASET in dataset_text and EXPECTED_SPLIT in dataset_text, "dataset/split hash registry")

print("PASS: compact canonical metadata and result tables are consistent")
print("INFO: weights are external artifacts; their recorded SHA-256 values were validated, not their absent bytes")
