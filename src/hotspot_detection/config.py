from dataclasses import dataclass, field


@dataclass(slots=True)
class HotspotConfig:
    base_metrics: list[str] = field(
        default_factory=lambda: ["throughput", "latency", "iops", "cpu"]
    )
    imbalance_metrics: list[str] = field(
        default_factory=lambda: [
            "imb_top1_share",
            "imb_norm_top1_share",
            "imb_hhi",
            "imb_gini",
            "imb_theil",
        ]
    )
    primary_hot_imbalance: str = "imb_norm_top1_share"
    k_sigma: float = 3.0
    baseline_fraction: float = 0.2
    min_n: int = 60
    persist_window: int = 3
    persist_min_hits: int = 2
    train_quantile: float = 0.70
    valid_quantile: float = 0.85


def default_config() -> HotspotConfig:
    return HotspotConfig()

