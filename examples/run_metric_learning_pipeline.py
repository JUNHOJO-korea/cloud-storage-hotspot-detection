from hotspot_detection.config import default_config


def main() -> None:
    cfg = default_config()
    print("Metric learning pipeline example")
    print(f"Default train quantile: {cfg.train_quantile}")
    print(f"Default valid quantile: {cfg.valid_quantile}")
    print("Full learning logic remains in the research notebooks and will be migrated into src/.")


if __name__ == "__main__":
    main()

