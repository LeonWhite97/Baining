from __future__ import annotations

from pathlib import Path

import yaml


DEFECT_NAMES: tuple[str, ...] = (
    "BALL_BRIDGE",
    "MISSING_BALL",
    "EXTRA_BALL",
    "BALL_SIZE_ABNORMAL",
    "BALL_OFFSET",
    "BALL_SHAPE_ABNORMAL",
    "FOREIGN_MATERIAL",
)
INPUT_CONTRACT = "rgb_grayscale_stack_v1"
REQUIRED_LIGHTS: tuple[str, ...] = ("R", "G", "B", "RING")


def load_class_names(path: Path) -> tuple[str, ...]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"CLASS_CONFIG_INVALID: {path.name}") from exc
    names = document.get("names") if isinstance(document, dict) else None
    if not isinstance(names, dict) or not names:
        raise ValueError("CLASS_CONFIG_INVALID: names must be a non-empty mapping")
    if set(names) != set(range(len(names))):
        raise ValueError("CLASS_CONFIG_INVALID: keys must be contiguous integers starting at zero")
    ordered = tuple(names[index] for index in range(len(names)))
    if any(not isinstance(name, str) or not name.strip() for name in ordered):
        raise ValueError("CLASS_CONFIG_INVALID: names must be non-empty strings")
    if len(set(ordered)) != len(ordered):
        raise ValueError("CLASS_CONFIG_INVALID: names must be unique")
    if path.name == "classes.yaml" and ordered != DEFECT_NAMES:
        raise ValueError("CLASS_CONFIG_INVALID: formal class order does not match the contract")
    return ordered
