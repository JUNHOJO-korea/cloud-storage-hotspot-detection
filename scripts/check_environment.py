import importlib


REQUIRED = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
]


def main() -> int:
    missing = []
    for module_name in REQUIRED:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if missing:
        print("Missing modules:")
        for item in missing:
            print(f"- {item}")
        return 1
    print("Environment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

