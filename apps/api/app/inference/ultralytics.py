from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from threading import Lock
from time import perf_counter_ns
from typing import Callable

from app.inference.base import Detection, InferenceOutput, InferenceRequest, InferenceUnavailable
from app.inference.preprocessing import stack_rgb_grayscale


DEFECT_NAMES = (
    "BALL_BRIDGE",
    "MISSING_BALL",
    "EXTRA_BALL",
    "BALL_SIZE_ABNORMAL",
    "BALL_OFFSET",
    "BALL_SHAPE_ABNORMAL",
    "FOREIGN_MATERIAL",
)
INPUT_CONTRACT = "rgb_grayscale_stack_v1"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_loader(path: Path):
    from ultralytics import YOLO

    return YOLO(str(path))


class UltralyticsInferenceAdapter:
    def __init__(
        self,
        *,
        model_path: Path,
        metadata_path: Path,
        device: str,
        imgsz: int,
        conf: float,
        model_loader: Callable[[Path], object] | None = None,
    ) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self._model_loader = model_loader or _default_loader
        self._model: object | None = None
        self._load_lock = Lock()
        self.model_version = "ultralytics:unvalidated"

    def _validate_package(self) -> dict[str, object]:
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError
            if tuple(metadata["names"]) != DEFECT_NAMES:
                raise ValueError
            if metadata["input_contract"] != INPUT_CONTRACT or metadata["task"] != "detect":
                raise ValueError
            if metadata["intended_use"] != "portfolio_internal_poc":
                raise ValueError
            if int(metadata["imgsz"]) != self.imgsz:
                raise ValueError
            if _sha256_file(self.model_path) != metadata["model_sha256"]:
                raise ValueError
            model_version = metadata["model_version"]
            if not isinstance(model_version, str) or not model_version:
                raise ValueError
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InferenceUnavailable("model package integrity validation failed") from exc
        self.model_version = model_version
        return metadata

    def _get_model(self) -> object:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        self._model = self._model_loader(self.model_path)
                    except Exception as exc:
                        raise InferenceUnavailable("model runtime could not be initialized") from exc
        return self._model

    @staticmethod
    def _rows(prediction: object) -> list[list[float]] | list[tuple[float, ...]]:
        if not isinstance(prediction, (list, tuple)) or len(prediction) != 1:
            raise ValueError("unexpected prediction batch")
        result = prediction[0]
        if hasattr(result, "boxes"):
            boxes = result.boxes
            return [] if boxes is None else boxes.data.tolist()
        if isinstance(result, (list, tuple)):
            return list(result)
        raise ValueError("unexpected prediction result")

    def predict(self, request: InferenceRequest) -> InferenceOutput:
        self._validate_package()
        started = perf_counter_ns()
        try:
            image = stack_rgb_grayscale(request.images)
            model = self._get_model()
            prediction = model.predict(
                source=image,
                imgsz=self.imgsz,
                conf=self.conf,
                device=self.device,
                save=False,
                verbose=False,
            )
            detections: list[Detection] = []
            for row in self._rows(prediction):
                if len(row) != 6:
                    raise ValueError("unexpected prediction row")
                x1, y1, x2, y2, confidence, class_id_value = (float(value) for value in row)
                class_id = int(class_id_value)
                if (
                    class_id_value != class_id
                    or not 0 <= class_id < len(DEFECT_NAMES)
                    or not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence))
                    or not 0 <= confidence <= 1
                    or x2 <= x1
                    or y2 <= y1
                ):
                    raise ValueError("invalid prediction row")
                detections.append(
                    Detection(
                        x=int(round(x1)),
                        y=int(round(y1)),
                        w=int(round(x2 - x1)),
                        h=int(round(y2 - y1)),
                        class_id=class_id,
                        defect_code=DEFECT_NAMES[class_id],
                        confidence=round(confidence, 8),
                    )
                )
        except InferenceUnavailable:
            raise
        except Exception as exc:
            raise InferenceUnavailable("model inference failed") from exc
        detections.sort(key=lambda item: item.confidence, reverse=True)
        primary = detections[0] if detections else None
        latency_ms = max(1, round((perf_counter_ns() - started) / 1_000_000))
        return InferenceOutput(
            model_version=self.model_version,
            normal_confidence=0.0,
            defect_score=primary.confidence if primary else 0.0,
            defect_code=primary.defect_code if primary else None,
            detections=tuple(detections),
            latency_ms=latency_ms,
        )
