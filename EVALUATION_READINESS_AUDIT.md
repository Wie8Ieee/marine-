# Evaluation readiness audit (read-only)

## Confirmed implementation details

| Check | Finding | Evidence | Readiness |
|---|---|---|---|
| Common confidence | `0.25` in all three canonical configs; evaluator uses `confidence_threshold`. | configs line 53; `marine_3model_experiment.py` | Ready |
| Matching IoU | `0.50` in all three canonical configs. | configs line 54 | Ready |
| River mapping | River loader uses one class and class-agnostic evaluation. | `evaluate_cross_domain` | Ready |
| River NMS | TorchVision uses the class-agnostic NMS context once per forward pass. | `class_agnostic_detection_nms_once` | Ready |
| FPS batch | FPS loaders use batch size 1. | detector and River evaluation paths | Ready |
| Warm-up | Timing function defaults to 20 warm-up frames and 120 timed frames. | `measure_torchvision_fps` | Ready |
| Same hardware | The protocol records hardware, but enforcement is operational rather than automatic. | `experiment_protocol.json` generation | Requires operator control |
| Test/River for best selection | Best checkpoint is updated from validation `map`; Test/River/FPS occur only in evaluation branches. | training loop and evaluators | Ready |

## Findings and recommendations

1. Run all final FPS measurements in one explicitly recorded hardware/software environment; compare neither FPS nor latency across heterogeneous devices.
2. Keep `evaluate: false` during canonical training. Run Trash Test, River, and timing only after artifact verification of `best` checkpoints.
3. Archive the resulting `python_environment.txt`, GPU/system details, config, hashes, and checkpoint SHA-256 alongside every evaluation.
4. The evaluation code is ready, but no canonical Test, River, or FPS result is currently approved for the manuscript.
