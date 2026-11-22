import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    groups = sorted({r["group"] for r in rows})
    if len(groups) != 2:
        raise SystemExit(f"need exactly 2 groups, got {groups}")
    return rows, groups


def column(rows, group, name):
    return np.array([float(r[name]) for r in rows if r["group"] == group and r.get(name, "") != ""])


def rank_biserial(a, b):
    return 2 * stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1


def bootstrap_ci(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def describe(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"median {med:.2f} (IQR {q1:.2f}-{q3:.2f}), n={len(x)}"


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
    rows, (a, b) = load(path)
    gain = {}
    for g in (a, b):
        pre, post = column(rows, g, "pre_mm"), column(rows, g, "post_mm")
        gain[g] = pre - post
        p = stats.wilcoxon(pre, post).pvalue if len(pre) > 1 and (pre != post).any() else 1.0
        print(f"{g}: before {describe(pre)}; after {describe(post)}; change within group p={p:.3f}")
    lo, hi = bootstrap_ci(gain[b], gain[a])
    print(f"improvement {a}: {describe(gain[a])}")
    print(f"improvement {b}: {describe(gain[b])}")
    print(f"{b} vs {a}: Mann-Whitney p={stats.mannwhitneyu(gain[b], gain[a]).pvalue:.3f}, "
          f"effect size r={rank_biserial(gain[b], gain[a]):.2f}, "
          f"difference of median improvement 95% CI [{lo:.2f}, {hi:.2f}] mm")
    if "post_breach" in rows[0]:
        table = [[int(column(rows, g, "post_breach").sum()), int(len(column(rows, g, "post_breach")) - column(rows, g, "post_breach").sum())] for g in (a, b)]
        print(f"breach after training: {a} {table[0][0]}/{sum(table[0])}, {b} {table[1][0]}/{sum(table[1])}, Fisher p={stats.fisher_exact(table).pvalue:.3f}")
    plt.figure(figsize=(6, 4))
    data = [column(rows, g, c) for g in (a, b) for c in ("pre_mm", "post_mm")]
    plt.boxplot(data, tick_labels=[f"{g}\nbefore" if c == "pre_mm" else f"{g}\nafter" for g in (a, b) for c in ("pre_mm", "post_mm")])
    plt.ylabel("entry point error, mm")
    plt.tight_layout()
    plt.savefig(path.with_name("stats.png"), dpi=120)
