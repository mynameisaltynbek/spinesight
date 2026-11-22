import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from net import Net, decode, errors, image_path, prep, solve_pose

ROOT = Path(__file__).resolve().parent.parent
TOL_MM = 3.0
TARGET_RISK = 0.05
SIGNALS = ["spread_mm", "low_confidence", "reprojection_px"]


def load_models(folder):
    nets, kps = [], None
    for p in sorted(folder.glob("model_s*.pt")):
        ck = torch.load(p, map_location="cpu")
        net = Net(len(ck["keypoints_3d"]), pretrained=False)
        net.load_state_dict(ck["state"])
        net.eval()
        nets.append(net)
        kps = np.array(ck["keypoints_3d"])
    if len(nets) < 2:
        raise SystemExit("need at least 2 models named model_s*.pt")
    return nets, kps


def spread(poses, screws):
    worst = 0.0
    for s in screws:
        e = np.array([np.array(s["entry"]) @ R.T + t for R, t in poses])
        worst = max(worst, float(np.linalg.norm(e - e.mean(0), axis=1).max()))
    return worst


def reprojection(R, t, kps, uv, K, dist):
    p, _ = cv2.projectPoints(kps, cv2.Rodrigues(R)[0], t, K, dist)
    return float(np.linalg.norm(p.reshape(-1, 2) - uv, axis=1).mean())


def predict(nets, img, kps, K, dist, screws):
    x, s = prep(img)
    with torch.no_grad():
        hms = [n(x[None])[0].numpy() for n in nets]
    poses = []
    for h in hms:
        uv, conf = decode(h)
        poses.append(solve_pose(uv / s, conf, kps, K, dist))
    uv, conf = decode(np.mean(hms, 0))
    uv /= s
    pose = solve_pose(uv, conf, kps, K, dist)
    return pose, {
        "spread_mm": spread(poses, screws) if pose and all(poses) else float("inf"),
        "low_confidence": float(1 - conf.mean()),
        "reprojection_px": reprojection(*pose, kps, uv, K, dist) if pose else float("inf"),
    }


def risk_coverage(signal, bad):
    order = np.argsort(signal, kind="stable")
    k = np.arange(1, len(bad) + 1)
    risk = np.cumsum(bad[order]) / k
    keep = np.isfinite(np.asarray(signal)[order])
    return k[keep] / len(bad), risk[keep], float(risk[keep].mean()) if keep.any() else 1.0


def pick_threshold(signal, bad, target):
    order = np.argsort(signal, kind="stable")
    s = np.asarray(signal)[order]
    risk = np.cumsum(bad[order]) / np.arange(1, len(bad) + 1)
    ok = np.flatnonzero((risk <= target) & np.isfinite(s))
    return float(s[ok.max()]) if len(ok) else float("-inf")


if __name__ == "__main__":
    mode, folder = sys.argv[1], Path(sys.argv[2])
    if mode not in ("calibrate", "test"):
        raise SystemExit("usage: python ensemble.py calibrate|test FOLDER")
    nets, kps = load_models(ROOT / "model")
    screws = json.load(open(ROOT / "outputs" / "plan.json"))["screws"]
    files = sorted(folder.glob("[0-9]*.json"))
    if (folder / "keypoints.json").exists():
        n = max(1, len(files) // 20)
        files = files[-2 * n:-n] if mode == "calibrate" else files[-n:]
    rows = []
    for f in files:
        label = json.load(open(f))
        pose, sig = predict(nets, cv2.imread(str(image_path(f))), kps, np.array(label["K"]), np.array(label["dist"]), screws)
        err = float("inf")
        if pose:
            e = errors(*pose, np.array(label["R"]), np.array(label["t"]), kps, screws)
            err = max(e[f"{s['side']}_entry_mm"] for s in screws)
        rows.append({"frame": f.stem, "entry_error_mm": err, **sig})
    with open(folder / f"ensemble_{mode}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    bad = np.array([r["entry_error_mm"] > TOL_MM for r in rows])
    print(f"{len(rows)} frames, {len(nets)} models, bad = entry error > {TOL_MM} mm or no pose: {100 * bad.mean():.1f}%")
    plt.figure(figsize=(6, 4))
    for name in SIGNALS:
        cov, risk, aurc = risk_coverage(np.array([r[name] for r in rows]), bad)
        plt.plot(cov, risk, label=f"{name} (AURC {aurc:.3f})")
        print(f"{name:16s} AURC {aurc:.3f}")
    plt.xlabel("coverage (share of frames answered)")
    plt.ylabel(f"risk (share with entry error > {TOL_MM} mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / f"risk_coverage_{mode}.png", dpi=120)

    spreads = np.array([r["spread_mm"] for r in rows])
    path = ROOT / "model" / "threshold.json"
    if mode == "calibrate":
        th = pick_threshold(spreads, bad, TARGET_RISK)
        json.dump({"spread_mm": th, "target_risk": TARGET_RISK, "tol_mm": TOL_MM}, open(path, "w"), indent=2)
    th = json.load(open(path))["spread_mm"]
    answered = spreads <= th
    risk = bad[answered].mean() if answered.any() else 0.0
    print(f"threshold spread <= {th:.2f} mm: answers {100 * answered.mean():.1f}% of frames, {100 * risk:.1f}% of answers bad")
