import numpy as np

from ensemble import pick_threshold, risk_coverage, spread

SCREWS = [{"side": "left", "entry": [0, 0, 0], "tip": [0, 40, 0]}, {"side": "right", "entry": [20, 0, 0], "tip": [20, 40, 0]}]


def test_spread():
    I = np.eye(3)
    assert spread([(I, np.zeros(3)), (I, np.zeros(3))], SCREWS) == 0
    assert np.isclose(spread([(I, np.zeros(3)), (I, np.array([0, 0, 4.0]))], SCREWS), 2)


def test_risk_coverage_and_threshold():
    signal = np.array([3.0, 1, 4, 2, np.inf])
    bad = np.array([True, False, True, False, True])
    cov, risk, _ = risk_coverage(signal, bad)
    assert np.allclose(cov, [0.2, 0.4, 0.6, 0.8]) and np.allclose(risk, [0, 0, 1 / 3, 0.5])
    assert pick_threshold(signal, bad, 0.0) == 2
    assert pick_threshold(signal, bad, 0.4) == 3
    assert pick_threshold(np.array([1.0]), np.array([True]), 0.0) == float("-inf")
