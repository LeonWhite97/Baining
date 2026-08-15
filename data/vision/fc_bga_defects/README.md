# FC-BGA Formal Dataset Contract

This directory is a structure-only template. Images and labels are intentionally ignored by Git.

Each source sample is one UTF-8 JSONL record:

```json
{"sample_id":"LOT01-TRAY02-A07","group_id":"LOT01","split":"train","images":{"R":"raw/R.png","G":"raw/G.png","B":"raw/B.png","RING":"raw/RING.png"},"label":"annotations/LOT01-TRAY02-A07.txt"}
```

`images` must contain exactly `R`, `G`, `B`, and `RING`. The converter turns R/G/B into grayscale channels and stacks them in that order. RING remains required provenance evidence but is not a model channel. All four frames must have identical dimensions.

Labels use normalized YOLO detection rows: `class_id center_x center_y width height`. An empty label is a normal sample; `NORMAL` is not a class. `MISSING_BALL` annotations require an approved expected-ball grid or equivalent product reference so that absence boxes are auditable. Use `group_id` to keep the same lot or other approved physical group out of multiple splits.

The fixed class order is defined in `tools/vision/fc_bga_yolo/configs/classes.yaml`. Real formal images and labels are not included in this repository.
