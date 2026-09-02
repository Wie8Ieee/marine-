# Trash-ICRA19 Data Integrity Report

- Dataset SHA-256: `5e0f560955eaf8ae4c517aa7eb80215f273dc976e9385f71e87d22f6ffa9e4cf`
- Split manifest SHA-256: `58825c29880ba62bd73f3d0b9b8f7e0199f957f71b020a118051ed144f92234a`
- Images/labels: 7684/7684
- Objects: 11061
- Classes: plastic=6370, bio=2417, rov=2274
- Sequences: 175
- Empty labels: 1
- Sequence overlap: 0
- Malformed/out-of-range boxes: 0
- Missing images or labels: 0

Verification command:

```text
python tools/verify_dataset.py --manifest manifests/trash_icra19/canonical_split_manifest.csv
```
