from dataclasses import dataclass

from app.domain.enums import ResultCategory


QUALITY_FAIL_BINS = frozenset(range(201, 206))
SYSTEM_BINS = frozenset({290, 291, 292, 293, 299})


class ResultValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class HandlerResult:
    aoi_decision: str
    result_category: ResultCategory
    aoi_bin: int
    defect_code: str | None
    requires_review: bool


def _fail(code: str, message: str) -> None:
    raise ResultValidationError(code, message)


def validate_handler_result(result: HandlerResult) -> HandlerResult:
    if result.result_category is ResultCategory.SYSTEM and result.aoi_decision != "REVIEW":
        _fail("INVALID_SYSTEM_RESULT", "SYSTEM results must use REVIEW decision")

    if result.aoi_decision == "PASS":
        if (
            result.result_category is not ResultCategory.QUALITY
            or result.aoi_bin != 100
            or result.defect_code is not None
            or result.requires_review
        ):
            _fail("INVALID_PASS_RESULT", "PASS requires QUALITY BIN 100 without a defect or review")
        return result

    if result.aoi_decision == "FAIL":
        if result.result_category is not ResultCategory.QUALITY or result.aoi_bin not in QUALITY_FAIL_BINS:
            _fail("INVALID_FAIL_RESULT", "FAIL requires a QUALITY BIN from 201 through 205")
        if not result.defect_code:
            _fail("DEFECT_CODE_REQUIRED", "FAIL results require a defect code")
        if result.requires_review:
            _fail("INVALID_FAIL_RESULT", "FAIL results cannot require review")
        return result

    if result.aoi_decision == "REVIEW":
        if not result.requires_review:
            _fail("REVIEW_FLAG_REQUIRED", "REVIEW results must set requires_review")
        if result.result_category is ResultCategory.QUALITY:
            if result.aoi_bin != 280:
                _fail("INVALID_REVIEW_BIN", "QUALITY review requires BIN 280")
        elif result.aoi_bin not in SYSTEM_BINS:
            _fail("INVALID_SYSTEM_BIN", "Unsupported SYSTEM BIN")
        return result

    _fail("INVALID_DECISION", f"Unsupported AOI decision: {result.aoi_decision}")
