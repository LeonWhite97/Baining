# Public FC-BGA Candidate Attribution

The files under `review/images/` are redistributed as an exact-deduplicated review subset of the following sources:

## BGA RAM Chips Detection

- Work: BGA RAM Chips Detection
- Creator/publisher: paween
- Source: https://universe.roboflow.com/paween/bga-ram-chips-detection-t3cqn
- Dataset version: 1
- Retrieved: 2026-08-16
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License URL: https://creativecommons.org/licenses/by/4.0/
- Legal-code snapshot: `LICENSE-CC-BY-4.0.txt`
- Legal-code SHA-256: `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`

Repository processing removed 17 exact SHA-256 duplicates from 73 source-image entries and renamed the retained JPEGs to content-derived sample identifiers. No image content was intentionally modified during candidate preparation.

## BGA-Balls

- Work: BGA-Balls
- Creator/publisher: kenshin-blirtz
- Source: https://universe.roboflow.com/kenshin-blirtz/bga-balls-3ihxj
- Dataset version: 2
- Retrieved: 2026-08-17
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License URL: https://creativecommons.org/licenses/by/4.0/
- Legal-code snapshot: `LICENSE-CC-BY-4.0.txt`
- Legal-code SHA-256: `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`

Repository processing exact-SHA-256 deduplicated 54 source-image entries from the BGA-Balls dataset. Images already present from the BGA RAM Chips source were detected by content hash and not duplicated. No image content was intentionally modified during candidate preparation.

## Common Terms

`review/candidates.jsonl` records each retained image's original filename, source group, source identifier, and full SHA-256 digest. The upstream `NG`, `OK`, and `Ball` labels are not redistributed as seven-class FC-BGA labels and are not mapped to formal defect classes. Every committed candidate remains `review_required` until a reviewer can establish a visible class and bounding box at native resolution.
