#!/usr/bin/env bash
set -euo pipefail

out=/workspace/persistent/marine_evaluations/final_eval_aed8bedd_20260909T192703Z
manifest="$out/evaluation_artifact_manifest.json"
jq -e '.status == "VERIFIED_COMPLETE" and .training_performed == false and .test_rerun == false and .fps_rerun == false' "$manifest" >/dev/null
while IFS=$'\t' read -r name expected; do
  actual="$(sha256sum "$out/$name" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "HASH_MISMATCH $name"
    exit 1
  fi
done < <(jq -r '.artifacts | to_entries[] | [.key, .value.sha256] | @tsv' "$manifest")
jq -e '.status == "PASS" and .before == .after and .test_hashes_before == .test_hashes_after' "$out/training_artifact_immutability.json" >/dev/null
echo FINAL_ARTIFACT_VERIFICATION_PASS
