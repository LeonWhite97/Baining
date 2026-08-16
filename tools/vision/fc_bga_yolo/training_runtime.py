from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Literal


StageName = Literal["A", "B0", "B1"]


@dataclass(frozen=True, slots=True)
class HardwareProbe:
    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    cuda_total_memory_bytes: int | None
    cpu_name: str


@dataclass(frozen=True, slots=True)
class ResourceDecision:
    status: Literal["ready", "skipped_resource", "skipped_runtime"]
    device: str | None
    projected_seconds: float | None
    reason: str


def probe_hardware(torch_module: object | None = None) -> HardwareProbe:
    if torch_module is None:
        import torch as torch_module

    cuda = getattr(torch_module, "cuda")
    available = bool(cuda.is_available())
    count = int(cuda.device_count()) if available else 0
    if available and count < 1:
        raise ValueError("CUDA_PROBE_INCONSISTENT")
    name = str(cuda.get_device_name(0)) if available else None
    memory = int(cuda.get_device_properties(0).total_memory) if available else None
    return HardwareProbe(
        torch_version=str(getattr(torch_module, "__version__")),
        cuda_available=available,
        cuda_device_count=count,
        cuda_device_name=name,
        cuda_total_memory_bytes=memory,
        cpu_name=platform.processor() or platform.machine() or "unknown",
    )


def select_stage_device(stage: StageName, probe: HardwareProbe) -> ResourceDecision:
    if stage not in {"A", "B0", "B1"}:
        raise ValueError("TRAINING_STAGE_INVALID")
    if probe.cuda_available:
        return ResourceDecision("ready", "0", None, "CUDA_SELECTED")
    if stage == "B1":
        return ResourceDecision("skipped_resource", None, None, "CUDA_REQUIRED_FOR_B1")
    return ResourceDecision("ready", "cpu", None, "CPU_FALLBACK")


def estimate_total_seconds(
    elapsed_seconds: float,
    *,
    completed_epochs: int,
    target_epochs: int,
) -> float:
    if elapsed_seconds <= 0 or completed_epochs <= 0 or target_epochs < completed_epochs:
        raise ValueError("CALIBRATION_VALUES_INVALID")
    return elapsed_seconds * target_epochs / completed_epochs
