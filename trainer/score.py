import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spine import inside, line


def grade(breach_mm):
    if breach_mm <= 0:
        return "A"
    for limit, g in ((2, "B"), (4, "C"), (6, "D")):
        if breach_mm < limit:
            return g
    return "E"


def tool_in_vertebra(tool, tip_m, vertebra):
    (Rm, tm), (R, t) = tool, vertebra
    tip_c = Rm @ tip_m + tm
    d_c = tip_c - (Rm @ np.array([0, 0, -1.0]) + tm)
    return R.T @ (tip_c - t), R.T @ d_c / np.linalg.norm(d_c)


def score(tip, d, screws, m, sp, dist, r=3.25):
    d = d / np.linalg.norm(d)
    s = min(screws, key=lambda s: np.linalg.norm(tip - np.array(s["entry"])))
    e, t = np.array(s["entry"]), np.array(s["tip"])
    u = (t - e) / np.linalg.norm(t - e)
    out = {
        "side": s["side"],
        "entry_mm": round(float(np.linalg.norm(tip - e)), 2),
        "angle_deg": round(float(np.degrees(np.arccos(np.clip(d @ u, -1, 1)))), 1),
        "breach_mm": None,
        "grade": "-",
    }
    start = next((tip + k * d for k in np.arange(0, 15, 0.5) if inside(m, sp, tip + k * d)), None)
    if start is None:
        return out
    clear = line(m, sp, dist, start, d, r)[3]
    out["breach_mm"] = round(float(max(0.0, -clear)), 2)
    out["grade"] = grade(out["breach_mm"])
    return out
