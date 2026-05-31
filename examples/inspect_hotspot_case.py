from hotspot_detection.config import default_config


def main() -> None:
    cfg = default_config()
    print("Inspect hotspot case example")
    print(f"Primary hotspot metric: {cfg.primary_hot_imbalance}")
    print(f"Base metrics: {', '.join(cfg.base_metrics)}")


if __name__ == "__main__":
    main()

