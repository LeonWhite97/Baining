import pytest

from app.domain.enums import PublishStatus, ResultCategory
from app.domain.handler_result import HandlerResult, ResultValidationError, validate_handler_result


@pytest.mark.parametrize(
    "result",
    [
        HandlerResult("PASS", ResultCategory.QUALITY, 100, None, False),
        HandlerResult("FAIL", ResultCategory.QUALITY, 201, "BALL_BRIDGE", False),
        HandlerResult("REVIEW", ResultCategory.QUALITY, 280, None, True),
        HandlerResult("REVIEW", ResultCategory.SYSTEM, 290, "INPUT_INVALID", True),
        HandlerResult("REVIEW", ResultCategory.SYSTEM, 299, None, True),
    ],
)
def test_valid_handler_result_combinations_are_accepted(result: HandlerResult) -> None:
    assert validate_handler_result(result) == result


@pytest.mark.parametrize(
    ("result", "expected_code"),
    [
        (HandlerResult("PASS", ResultCategory.QUALITY, 201, None, False), "INVALID_PASS_RESULT"),
        (HandlerResult("PASS", ResultCategory.QUALITY, 100, "BALL_BRIDGE", False), "INVALID_PASS_RESULT"),
        (HandlerResult("FAIL", ResultCategory.QUALITY, 201, None, False), "DEFECT_CODE_REQUIRED"),
        (HandlerResult("FAIL", ResultCategory.SYSTEM, 291, "MODEL_UNAVAILABLE", True), "INVALID_SYSTEM_RESULT"),
        (HandlerResult("REVIEW", ResultCategory.QUALITY, 280, None, False), "REVIEW_FLAG_REQUIRED"),
        (HandlerResult("REVIEW", ResultCategory.SYSTEM, 294, None, True), "INVALID_SYSTEM_BIN"),
    ],
)
def test_inconsistent_handler_results_are_rejected(result: HandlerResult, expected_code: str) -> None:
    with pytest.raises(ResultValidationError) as exc_info:
        validate_handler_result(result)

    assert exc_info.value.code == expected_code


def test_publish_status_contains_fail_closed_delivery_states() -> None:
    assert PublishStatus.NOT_READY.value == "NOT_READY"
    assert PublishStatus.PENDING.value == "PENDING"
    assert PublishStatus.DELIVERY_UNCERTAIN.value == "DELIVERY_UNCERTAIN"
    assert PublishStatus.EXPIRED.value == "EXPIRED"
