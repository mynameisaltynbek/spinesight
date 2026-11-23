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


def test_summarize_means_and_breach():
    from summarize import summarize
    row = lambda p, g, ph, mm, gr, sb="0": {"participant": p, "group": g, "phase": ph, "entry_mm": mm, "grade": gr, "sensor_breach": sb}
    rows = [row("P1", "control", "pre", "6", "A"), row("P1", "control", "pre", "4", "C"), row("P1", "control", "train", "9", "E"),
            row("P1", "control", "post", "3", "B", "1"), row("P2", "spinesight", "pre", "5", "A")]
    out = summarize(rows)
    assert out == [{"participant": "P1", "group": "control", "pre_mm": 5.0, "post_mm": 3.0, "pre_breach": 1, "post_breach": 1}]
