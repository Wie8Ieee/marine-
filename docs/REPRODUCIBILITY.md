# Reproducibility

Use the pinned configurations in `configs/`, verify the dataset and sequence-safe split identities in `provenance/`, and run the source under `src/` with the pinned CUDA environment in `requirements.txt`. Model weights are external because GitHub is not used for large checkpoints. Verify externally supplied weights against `provenance/checkpoint_hashes.txt`, then run `python scripts/verify_artifacts.py`.

Training and evaluation must use separate outputs. Test, River, and FPS must not influence checkpoint selection.
