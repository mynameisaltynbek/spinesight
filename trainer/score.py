import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spine import inside, line, walk


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
        "wall": "",
        "depth_mm": None,
    }
    start = next((tip + k * d for k in np.arange(0, 15, 0.5) if inside(m, sp, tip + k * d)), None)
    if start is None:
        return out
    entry, _, length, clear = line(m, sp, dist, start, d, r)
    out["breach_mm"] = round(float(max(0.0, -clear)), 2)
    out["grade"] = grade(out["breach_mm"])
    if clear < 0:
        reach = (walk(m, sp, start, d) - start) @ d
        ks = np.arange(5, reach, 0.5) if reach > 5.5 else np.array([0.0])
        k = min(ks, key=lambda k: dist[tuple(np.round((start + k * d) / sp).astype(int))])
        q = start + max(k, 10) * d
        v = q - e - ((q - e) @ u) * u
        medial = (1 if s["side"] == "left" else -1) * v[0]
        if abs(v[0]) >= abs(v[2]):
            out["wall"] = "medial" if medial > 0 else "lateral"
        else:
            out["wall"] = "cranial" if v[2] > 0 else "caudal"
        out["depth_mm"] = round(float((start - entry) @ d + k), 1)
    return out
