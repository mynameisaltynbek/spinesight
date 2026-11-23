import csv
import sys
from pathlib import Path

import numpy as np


def bad(row):
    return row["grade"] in ("C", "D", "E") or row["sensor_breach"] == "1"


def summarize(rows):
    out = []
    for p in sorted({r["participant"] for r in rows}):
        mine = [r for r in rows if r["participant"] == p]
        groups = {r["group"] for r in mine}
        if len(groups) != 1:
            raise SystemExit(f"{p} is in more than one group: {groups}")
        pre = [r for r in mine if r["phase"] == "pre"]
        post = [r for r in mine if r["phase"] == "post"]
        if not pre or not post:
            print(f"skipped {p}: needs both pre and post attempts")
            continue
        out.append({
            "participant": p,
            "group": groups.pop(),
            "pre_mm": round(float(np.mean([float(r["entry_mm"]) for r in pre])), 2),
            "post_mm": round(float(np.mean([float(r["entry_mm"]) for r in post])), 2),
            "pre_breach": int(any(bad(r) for r in pre)),
            "post_breach": int(any(bad(r) for r in post)),
        })
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "attempts.csv")
    rows = summarize(list(csv.DictReader(open(src, encoding="utf-8"))))
    with open(src.with_name("results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} participants written to results.csv; next: python stats.py results.csv")
