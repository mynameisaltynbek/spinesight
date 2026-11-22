import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from net import Net, decode, errors, image_path, keypoints_2d, prep, solve_pose

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "rig"))
from common import load_stl

ckpt = torch.load(sys.argv[1] if len(sys.argv) > 1 else ROOT / "model" / "model_s0.pt", map_location="cpu")
folder = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "test_real"
kps = np.array(ckpt["keypoints_3d"])
net = Net(len(kps), pretrained=False)
net.load_state_dict(ckpt["state"])
net.eval()
pts = load_stl(ROOT / "outputs" / "vertebra.stl").reshape(-1, 3)[::10]
diameter = max(np.linalg.norm(pts - p, axis=1).max() for p in pts[::20])
screws = json.load(open(ROOT / "outputs" / "plan.json"))["screws"]

files = sorted(folder.glob("[0-9]*.json"))
if (folder / "keypoints.json").exists():
    files = files[-max(1, len(files) // 20):]
rows, failed = [], 0
for f in files:
    label = json.load(open(f))
    x, s = prep(cv2.imread(str(image_path(f))))
    with torch.no_grad():
        uv, conf = decode(net(x[None])[0].numpy())
    uv /= s
    pose = solve_pose(uv, conf, kps, np.array(label["K"]), np.array(label["dist"]))
    if pose is None:
        failed += 1
        continue
    e = errors(*pose, np.array(label["R"]), np.array(label["t"]), pts, screws)
    e["keypoint_px"] = float(np.linalg.norm(uv - keypoints_2d(label, kps), axis=1).mean())
    rows.append({"frame": f.stem, **e})

if not rows:
    raise SystemExit(f"no pose found in any of {len(files)} frames")
with open(folder / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(files)} frames, {failed} failed (no pose)")
for key in rows[0]:
    if key != "frame":
        v = np.array([r[key] for r in rows])
        print(f"{key:18s} median {np.median(v):8.2f}   90% {np.percentile(v, 90):8.2f}")
add = np.array([r["add_mm"] for r in rows])
print(f"ADD < 10% of size ({0.1 * diameter:.1f} mm): {100 * (add < 0.1 * diameter).sum() / len(files):.1f}% of all frames")
