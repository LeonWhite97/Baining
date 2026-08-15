from app.domain.enums import AssociationStatus, Decision
from app.domain.source_key import SourceKeyParts, generate_source_key_hash


def test_source_key_is_stable_and_light_id_is_attachment_only() -> None:
    parts = SourceKeyParts(
        device_id="PIS-01",
        device_session_id="BOOT-20240901T080000Z",
        inspection_sequence="1042",
        tray_id="TRAY-09",
        slot_index="A07",
        surface="TOP",
    )

    assert generate_source_key_hash(parts) == (
        "59d3967524fe44d8f7141c74e8083df90e08a72e5bc46c2be718f135679ce8c5"
    )
    assert generate_source_key_hash(parts) == generate_source_key_hash(parts)


def test_device_session_separates_restarted_counter() -> None:
    first_boot = SourceKeyParts("PIS-01", "BOOT-A", "1", "T-01", "01", "TOP")
    second_boot = SourceKeyParts("PIS-01", "BOOT-B", "1", "T-01", "01", "TOP")

    assert generate_source_key_hash(first_boot) != generate_source_key_hash(second_boot)


def test_domain_enums_expose_safe_terminal_decisions() -> None:
    assert Decision.PASS.value == "PASS"
    assert Decision.FAIL.value == "FAIL"
    assert Decision.REVIEW.value == "REVIEW"
    assert AssociationStatus.QUARANTINED.is_terminal is True
    assert AssociationStatus.READY.is_terminal is False
