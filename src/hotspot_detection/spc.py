import numpy as np


def robust_mad_sigma(values: list[float] | np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    sigma = float(1.4826 * mad)
    if not np.isfinite(sigma) or sigma < 1e-12:
        sigma = float(np.std(x))
    if not np.isfinite(sigma) or sigma < 1e-12:
        sigma = max(abs(median) * 0.01, 1e-3)
    return median, sigma


def fit_upper_control_limit(
    values: list[float] | np.ndarray,
    *,
    baseline_fraction: float = 0.2,
    min_n: int = 60,
    k_sigma: float = 3.0,
) -> dict[str, float] | None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < max(10, min_n):
        return None
    baseline_n = min(int(max(min_n, round(x.size * baseline_fraction))), x.size)
    baseline = x[:baseline_n]
    cl, sigma = robust_mad_sigma(baseline)
    return {
        "CL": float(cl),
        "SIGMA": float(sigma),
        "UCL": float(cl + (k_sigma * sigma)),
        "LCL": float(cl - (k_sigma * sigma)),
        "baseline_n": float(baseline_n),
    }


def persistence_event(hit: list[int] | np.ndarray, window: int, min_hits: int) -> np.ndarray:
    x = np.asarray(hit, dtype=int)
    if window <= 1:
        return x
    out = np.zeros_like(x)
    for idx in range(window - 1, len(x)):
        out[idx] = int(np.sum(x[idx - window + 1 : idx + 1]) >= min_hits)
    return out

