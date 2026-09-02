from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from marine_3model_experiment import build_faster_rcnn, build_ssdlite, letterbox_image_and_boxes

DATA = ROOT / "extracted_marine_3model_comparison/marine_3model_comparison/prepared_datasets/trash_icra19_clean"
IMAGE_STEM = os.environ.get("QUAL_IMAGE_STEM", "obj0002_frame0000009_88385004")
FRCNN_CKPT = ROOT / ".kaggle_training_progress_latest/marine_3model_comparison/runs/seed_123/torchvision/frcnn/best.pt"
SSD_CKPT = ROOT / ".kaggle_ssd_final/marine_3model_comparison/runs/seed_123/torchvision/ssd/best.pt"
YOLO_CKPT = ROOT / "training_results/yolov8s_seed_123/marine_3model_comparison/runs/seed_123/yolo/yolov8s_stage1/weights/best.pt"
OUT = ROOT / "research_figures/publication_ready"
OUTPUT_BASENAME = os.environ.get("QUAL_OUTPUT_BASENAME", "figure_qualitative_model_comparison_clean")

CLASSES = ["plastic", "bio", "rov"]
COLORS = {"plastic": "#ef3b2c", "bio": "#24a148", "rov": "#1982c4"}
DISPLAY_THRESHOLD = 0.50


def image_path() -> Path:
    found = list((DATA / "images/test").glob(IMAGE_STEM + ".*"))
    if not found:
        raise FileNotFoundError(IMAGE_STEM)
    return found[0]


def torchvision_predictions(model, checkpoint: Path, image: Image.Image, size: int):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    boxed, _ = letterbox_image_and_boxes(image, np.empty((0, 4), dtype=np.float32), size)
    tensor = torch.from_numpy(np.asarray(boxed).astype(np.float32) / 255.0).permute(2, 0, 1)
    with torch.inference_mode():
        pred = model([tensor])[0]

    ow, oh = image.size
    scale = min(size / ow, size / oh)
    nw, nh = round(ow * scale), round(oh * scale)
    px, py = (size - nw) // 2, (size - nh) // 2
    result = []
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        score = float(score)
        label = int(label) - 1
        if score < DISPLAY_THRESHOLD or not 0 <= label < len(CLASSES):
            continue
        x1, y1, x2, y2 = box.tolist()
        result.append({
            "box": [(x1-px)/scale, (y1-py)/scale, (x2-px)/scale, (y2-py)/scale],
            "class": CLASSES[label], "score": score,
        })
    return result


def yolo_predictions(image_file: Path):
    from ultralytics import YOLO
    model = YOLO(str(YOLO_CKPT))
    prediction = model.predict(str(image_file), conf=DISPLAY_THRESHOLD, verbose=False, device="cpu")[0]
    result = []
    if prediction.boxes is None:
        return result
    for box, cls, score in zip(prediction.boxes.xyxy.cpu(), prediction.boxes.cls.cpu(), prediction.boxes.conf.cpu()):
        label = int(cls)
        if 0 <= label < len(CLASSES):
            result.append({"box": box.tolist(), "class": CLASSES[label], "score": float(score)})
    return result


def draw_panel(ax, image, title, predictions=None):
    ax.imshow(image)
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=9)
    ax.axis("off")
    for pred in predictions or []:
        x1, y1, x2, y2 = pred["box"]
        color = COLORS[pred["class"]]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2.4)
        ax.add_patch(rect)
        ax.text(x1, max(4, y1-5), f'{pred["class"]} {pred["score"]:.2f}', color="white",
                fontsize=9, fontweight="bold", va="bottom",
                bbox=dict(facecolor=color, edgecolor="none", pad=2.2, alpha=0.95))


def main():
    source = image_path()
    image = Image.open(source).convert("RGB")
    predictions = {
        "YOLOv8s": yolo_predictions(source),
        "Faster R-CNN": torchvision_predictions(build_faster_rcnn(4), FRCNN_CKPT, image, 640),
        "MobileNet SSD": torchvision_predictions(build_ssdlite(4), SSD_CKPT, image, 320),
    }
    for name, preds in predictions.items():
        print(name, [(p["class"], round(p["score"], 3), [round(v, 1) for v in p["box"]]) for p in preds])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.7), constrained_layout=False)
    fig.patch.set_facecolor("white")
    draw_panel(axes[0], image, "Original test image")
    for ax, (name, preds) in zip(axes[1:], predictions.items()):
        draw_panel(ax, image, f"{name}  |  {len(preds)} detection{'s' if len(preds) != 1 else ''}", preds)
    fig.suptitle("Three-model qualitative comparison on Trash-ICRA19", fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.5, 0.035, f"Displayed predictions: confidence ≥ {DISPLAY_THRESHOLD:.2f}  •  Class color: plastic",
             ha="center", fontsize=10, color="#444444")
    plt.subplots_adjust(left=0.025, right=0.985, bottom=0.10, top=0.87, wspace=0.025)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{OUTPUT_BASENAME}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{OUTPUT_BASENAME}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
