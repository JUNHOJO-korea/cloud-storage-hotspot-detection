from dataclasses import dataclass


BASE_LONG_COLUMNS = [
    "timestamp",
    "stg_ip",
    "level",
    "entity_id",
    "rw",
    "metric",
    "value",
]


@dataclass(frozen=True, slots=True)
class DataContract:
    name: str
    required_columns: tuple[str, ...]


BASE_LONG_CONTRACT = DataContract(
    name="base_long",
    required_columns=tuple(BASE_LONG_COLUMNS),
)

