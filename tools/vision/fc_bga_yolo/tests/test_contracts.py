from pathlib import Path

import pytest

from tools.vision.fc_bga_yolo.contracts import (
    DEFECT_NAMES,
    INPUT_CONTRACT,
    REQUIRED_LIGHTS,
    load_class_names,
)


def test_formal_class_order_is_stable() -> None:
    assert DEFECT_NAMES == (
        "BALL_BRIDGE",
        "MISSING_BALL",
        "EXTRA_BALL",
        "BALL_SIZE_ABNORMAL",
        "BALL_OFFSET",
        "BALL_SHAPE_ABNORMAL",
        "FOREIGN_MATERIAL",
    )
    assert INPUT_CONTRACT == "rgb_grayscale_stack_v1"
    assert REQUIRED_LIGHTS == ("R", "G", "B", "RING")


def test_classes_yaml_matches_python_contract() -> None:
    path = Path("tools/vision/fc_bga_yolo/configs/classes.yaml")
    assert load_class_names(path) == DEFECT_NAMES


def test_duplicate_class_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text("names:\n  0: BALL_BRIDGE\n  1: BALL_BRIDGE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_class_names(path)

