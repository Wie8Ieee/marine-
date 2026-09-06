# Post-smoke runbook

This runbook does not authorize official training automatically.

1. Wait for the active exact-resume smoke to finish. Download only its new output directory.
2. If it fails, preserve `process_b_stdout.log`, `process_b_stderr.log`, `process_b_execution.json`, and `process_b_failure.json`; diagnose before changing any launcher or canonical code.
3. If it succeeds, verify the smoke manifest, Process A contract, Process B resume provenance, history continuity, and reference comparison. Mark it **SMOKE_DEBUG_ONLY** and not research-eligible.
4. Request separate explicit authorization before launching Faster R-CNN Session A. Do not start Test, River, FPS, or Session B automatically.
5. After each official canonical training completes, verify `best`, `last`, history, runtime config, environment, fingerprints, and checkpoint hashes.
6. After all three approved checkpoint packages exist, request authorization for separate Trash Test, River, and same-hardware FPS evaluations.
7. Populate `FINAL_RESULTS_TEMPLATES.md` only from verified outputs. For multi-seed claims, complete the planned repeated runs before reporting mean ± SD or 95% CI.
