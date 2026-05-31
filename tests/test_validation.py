from hotspot_detection.schemas import BASE_LONG_CONTRACT
from hotspot_detection.validation import validate_columns


def test_validate_columns_detects_missing_values() -> None:
    missing = validate_columns(
        ["timestamp", "stg_ip", "level", "entity_id", "rw", "metric"],
        BASE_LONG_CONTRACT,
    )
    assert missing == ["value"]

