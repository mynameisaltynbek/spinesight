import numpy as np

from stats import bootstrap_ci, rank_biserial


def test_effect_size_sign_and_range():
    a = np.array([1.0, 2, 3, 4, 5])
    assert rank_biserial(a + 10, a) == 1
    assert rank_biserial(a, a + 10) == -1
    assert rank_biserial(a, a) == 0


def test_bootstrap_ci_contains_true_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(2, 1, 40)
    lo, hi = bootstrap_ci(a + 3, a, n=2000)
    assert lo < 3 < hi and lo > 0
