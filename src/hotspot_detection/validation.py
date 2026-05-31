from .schemas import DataContract


def validate_columns(columns: list[str], contract: DataContract) -> list[str]:
    missing = [col for col in contract.required_columns if col not in columns]
    return missing

