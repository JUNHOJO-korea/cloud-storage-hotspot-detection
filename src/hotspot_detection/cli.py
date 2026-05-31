import argparse
from dataclasses import asdict

from .config import default_config


def notebook_catalog() -> list[dict[str, str]]:
    return [
        {
            "name": "sample2.ipynb",
            "role": "main end-to-end research pipeline",
        },
        {
            "name": "sample.ipynb",
            "role": "earlier pipeline variant with node and port blocks",
        },
        {
            "name": "case experiment.ipynb",
            "role": "feature-family case screening notebook",
        },
    ]


def _show_config() -> int:
    cfg = default_config()
    for key, value in asdict(cfg).items():
        print(f"{key}: {value}")
    return 0


def _describe_notebooks() -> int:
    for item in notebook_catalog():
        print(f"{item['name']}: {item['role']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hotspot-detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-config")
    subparsers.add_parser("describe-notebooks")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "show-config":
        return _show_config()
    if args.command == "describe-notebooks":
        return _describe_notebooks()
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

