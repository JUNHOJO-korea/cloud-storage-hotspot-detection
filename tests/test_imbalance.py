import pytest

from hotspot_detection.imbalance import gini, hhi, normalized_top1_share, theil, top1_share


def test_balanced_distribution_metrics() -> None:
    values = [10, 10, 10, 10]
    assert gini(values) == pytest.approx(0.0)
    assert hhi(values) == pytest.approx(0.25)
    assert top1_share(values) == pytest.approx(0.25)
    assert normalized_top1_share(values) == pytest.approx(0.0)
    assert theil(values) == pytest.approx(0.0)


def test_concentrated_distribution_metrics() -> None:
    values = [100, 0, 0, 0]
    assert gini(values) > 0.7
    assert hhi(values) == pytest.approx(1.0)
    assert top1_share(values) == pytest.approx(1.0)
    assert normalized_top1_share(values) == pytest.approx(1.0)

