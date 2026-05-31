from hotspot_detection.cli import notebook_catalog


def main() -> None:
    print("Rule-based pipeline example")
    print("Current full implementation lives in notebooks.")
    for item in notebook_catalog():
        print(f"- {item['name']}: {item['role']}")


if __name__ == "__main__":
    main()

