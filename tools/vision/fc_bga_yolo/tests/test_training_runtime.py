from types import SimpleNamespace

import pytest

from tools.vision.fc_bga_yolo.training_runtime import (
    HardwareProbe,
    estimate_total_seconds,
    probe_hardware,
    select_stage_device,
)


CPU_PROBE = HardwareProbe(
    torch_version="2.13.0+cpu",
    cuda_available=False,
    cuda_device_count=0,
    cuda_device_name=None,
    cuda_total_memory_bytes=None,
    cpu_name="test-cpu",
)


def test_cpu_allows_stage_a_and_b0_but_skips_b1() -> None:
    assert select_stage_device("A", CPU_PROBE).device == "cpu"
    assert select_stage_device("B0", CPU_PROBE).device == "cpu"
    b1 = select_stage_device("B1", CPU_PROBE)
    assert b1.status == "skipped_resource"
    assert b1.device is None
    assert b1.reason == "CUDA_REQUIRED_FOR_B1"


def test_cuda_probe_selects_device_zero_for_every_stage() -> None:
    probe = HardwareProbe(
        torch_version="2.13.0+cu130",
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_name="Test GPU",
        cuda_total_memory_bytes=8 * 1024**3,
        cpu_name="test-cpu",
    )

    assert [select_stage_device(stage, probe).device for stage in ("A", "B0", "B1")] == ["0", "0", "0"]


def test_three_epoch_calibration_estimates_final_total_epochs() -> None:
    assert estimate_total_seconds(900.0, completed_epochs=3, target_epochs=30) == 9000.0
    with pytest.raises(ValueError, match="CALIBRATION_VALUES_INVALID"):
        estimate_total_seconds(0.0, completed_epochs=3, target_epochs=30)


def test_probe_hardware_reads_cuda_device_evidence() -> None:
    properties = SimpleNamespace(total_memory=8 * 1024**3)
    fake_torch = SimpleNamespace(
        __version__="2.13.0+cu130",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: "Test GPU",
            get_device_properties=lambda index: properties,
        ),
    )

    probe = probe_hardware(fake_torch)

    assert probe.cuda_device_count == 1
    assert probe.cuda_device_name == "Test GPU"
    assert probe.cuda_total_memory_bytes == 8 * 1024**3
