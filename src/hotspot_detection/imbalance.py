import numpy as np


def _clean(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return np.clip(arr, 0.0, None)


def gini(values: list[float] | np.ndarray) -> float:
    x = _clean(values)
    total = x.sum()
    if x.size < 2 or total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2 * np.sum(index * x) / (n * total)) - ((n + 1) / n))


def hhi(values: list[float] | np.ndarray) -> float:
    x = _clean(values)
    total = x.sum()
    if x.size < 2 or total <= 0:
        return 0.0
    p = x / total
    return float(np.sum(p**2))


def theil(values: list[float] | np.ndarray) -> float:
    x = _clean(values)
    total = x.sum()
    if x.size < 2 or total <= 0:
        return 0.0
    mean = total / x.size
    if mean <= 0:
        return 0.0
    y = x[x > 0]
    if y.size < 2:
        return 0.0
    z = y / mean
    return float(np.mean(z * np.log(z)))


def top1_share(values: list[float] | np.ndarray) -> float:
    x = _clean(values)
    total = x.sum()
    if x.size < 2 or total <= 0:
        return 0.0
    return float(np.max(x) / total)


def normalized_top1_share(values: list[float] | np.ndarray) -> float:
    x = _clean(values)
    total = x.sum()
    n = x.size
    if n < 2 or total <= 0:
        return 0.0
    raw = float(np.max(x) / total)
    uniform_floor = 1.0 / n
    denom = 1.0 - uniform_floor
    if denom <= 0:
        return 0.0
    return float(max(0.0, (raw - uniform_floor) / denom))

