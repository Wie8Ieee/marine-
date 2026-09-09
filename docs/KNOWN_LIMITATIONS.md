# Known limitations

- Only Seed 42 is reported; between-seed variability and confidence intervals require future independent runs.
- Input resolution is architecture-specific (640/640/320), which can affect small-object detection and speed.
- River evaluation measures zero-shot cross-domain transfer and does not represent training on River data.
- FPS is architecture-specific throughput on one NVIDIA A40 environment, not compute-normalized performance.
- Weights and raw datasets are external to this compact Git repository.
