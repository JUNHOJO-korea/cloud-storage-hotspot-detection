import pytest

from hotspot_detection.spc import fit_upper_control_limit, persistence_event, robust_mad_sigma


def test_robust_mad_sigma_returns_positive_sigma() -> None:
    center, sigma = robust_mad_sigma([1, 1, 1, 2, 2, 100])
    assert center == pytest.approx(1.5)
    assert sigma > 0


def test_fit_upper_control_limit_returns_limits() -> None:
    values = list(range(100))
    limits = fit_upper_control_limit(values, baseline_fraction=0.2, min_n=10, k_sigma=3.0)
    assert limits is not None
    assert limits["UCL"] > limits["CL"]


def test_persistence_event() -> None:
    hit = [0, 1, 1, 0, 1]
    event = persistence_event(hit, window=3, min_hits=2)
    assert event.tolist() == [0, 0, 1, 1, 1]

