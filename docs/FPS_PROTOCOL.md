# FPS protocol

- GPU: NVIDIA A40
- Batch size: 1; precision: FP32
- Warm-up: 20 frames; timed frames: 120
- Input: YOLOv8s 640, Faster R-CNN 640, SSDLite 320
- Decoding and external resizing were excluded; host-to-device transfer, forward pass, and detector-native postprocessing/NMS were included.
- CUDA was synchronized before and after the timed section. Model and dataset loading were excluded.

FPS values represent architecture-specific inference throughput. They are not a compute-normalized comparison.
