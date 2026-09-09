# Experiment protocol

Trash-ICRA19 was fixed by the recorded dataset and split fingerprints. The sequence-safe split contains 5,377 training, 1,154 validation, and 1,153 test images with zero sequence overlap. All models used Seed 42, horizontal flip probability 0.5, and no brightness/saturation augmentation; YOLO Mosaic, MixUp, and HSV augmentation were disabled. Each model used 10 frozen-head epochs followed by 100 full-fine-tuning epochs. YOLOv8s and Faster R-CNN used 640×640 inputs; SSDLite used its architecture-native 320×320 input.

The best checkpoint was selected only by validation mAP@0.5:0.95. Test evaluation was performed after selection. River was a zero-shot, class-agnostic cross-domain evaluation. AP/mAP used the complete ranked prediction set (`conf=0.0` at the TorchMetrics boundary); confidence 0.25 was used only for operational Precision, Recall, and F1. IoU matching was 0.50.
