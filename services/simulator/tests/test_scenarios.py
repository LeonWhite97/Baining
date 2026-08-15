from simulator.scenarios import ScenarioKind, build_scenario


def test_same_seed_builds_same_source_identity() -> None:
    first = build_scenario(42, ScenarioKind.DEFECT)
    second = build_scenario(42, ScenarioKind.DEFECT)

    assert first == second
    assert first["inspection_sequence"] == "42"
    assert first["scenario"] == "DEFECT"


def test_missing_light_scenario_remains_explicit() -> None:
    payload = build_scenario(7, ScenarioKind.MISSING_LIGHT)

    assert payload["scenario"] == "MISSING_LIGHT"
    assert payload["device_session_id"] == "SIM-BOOT-202408"
