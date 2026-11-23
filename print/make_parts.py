import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rig"))
sys.path.insert(0, str(ROOT / "synth"))
from common import load_stl
from generate import render
from spine import load, write_stl

STEP = 0.5
PLA = 1.24


def solid(m, sp):
    return ndimage.zoom(ndimage.gaussian_filter(m.astype(float), 1), sp / STEP, order=1) > 0.5


def surface(vox, step):
    v, f, _, _ = measure.marching_cubes(np.pad(vox, 2).astype(float), 0.5, spacing=(step,) * 3)
    return v - 2 * step, f


def socket(vox, depth=20.0, floor=3.0, margin=5.0, gap=1):
    z = np.flatnonzero(vox.any(axis=(0, 1)))[0]
    top = z + int(depth / STEP)
    cavity = np.logical_or.accumulate(ndimage.binary_dilation(vox, iterations=gap)[:, :, z:top], axis=2)
    xy = np.argwhere(cavity.any(axis=2))
    m = int(margin / STEP)
    lo, hi = xy.min(0) - m, xy.max(0) + m + 1
    padded = np.pad(cavity, ((m, m), (m, m), (0, 0)))[lo[0] + m:hi[0] + m, lo[1] + m:hi[1] + m]
    box = np.ones((padded.shape[0], padded.shape[1], int(floor / STEP) + padded.shape[2]), bool)
    box[:, :, int(floor / STEP):] = ~padded
    return box


def marker_holder(size=40.0, thick=2.0, hole=4.3, outer=9.0, length=25.0, step=0.25):
    x = (np.arange(int(size / step)) + 0.5) * step - size / 2
    y = (np.arange(int(outer / step)) + 0.5) * step - outer / 2
    z = (np.arange(int((length + size) / step)) + 0.5) * step
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.hypot(X, Y)
    return ((r <= outer / 2) & (r >= hole / 2) & (Z < length)) | ((np.abs(Y) <= thick / 2) & (Z >= length)), step


def measure_mesh(tris):
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum() / 2
    volume = abs(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum()) / 6
    return area, volume


def grams(area, volume, walls=1.2, infill=0.2):
    shell = min(area * walls, volume)
    return (shell + infill * (volume - shell)) / 1000 * PLA


def preview(tris, path):
    img = np.full((450, 600, 3), 255, np.uint8)
    K = np.array([[800.0, 0, 300], [0, 800, 225], [0, 0, 1]])
    R = cv2.Rodrigues(np.array([-2.1, 0.0, 0.6]))[0]
    v = tris.reshape(-1, 3)
    t = np.array([0, 0, np.ptp(v, 0).max() * 800 / 330]) - R @ v.mean(0)
    render(img, tris, R, t, K, np.array([200.0, 200, 200]), np.array([0.3, -0.5, -0.8]) / np.linalg.norm([0.3, -0.5, -0.8]), 0.2)
    cv2.imwrite(str(path), img)


if __name__ == "__main__":
    out = ROOT / "print" / "parts"
    out.mkdir(parents=True, exist_ok=True)
    m, sp = load(ROOT / "data" / "sub-verse031_seg-vert_msk.nii.gz", 22)
    shutil.copy(ROOT / "outputs" / "vertebra.stl", out / "vertebra_L3.stl")
    v, f = surface(socket(solid(m, sp)), STEP)
    write_stl(out / "socket_L3.stl", [(v, f)])
    vox, step = marker_holder()
    v, f = surface(vox, step)
    write_stl(out / "marker_holder.stl", [(v, f)])
    report = {}
    for name in ("vertebra_L3", "socket_L3", "marker_holder"):
        tris = load_stl(out / f"{name}.stl")
        area, volume = measure_mesh(tris)
        report[name] = {
            "size_mm": np.ptp(tris.reshape(-1, 3), 0).round(1).tolist(),
            "volume_cm3": round(volume / 1000, 1),
            "pla_grams_20pct_infill": round(grams(area, volume), 1),
            "pla_grams_solid": round(volume / 1000 * PLA, 1),
        }
        preview(tris, out / f"{name}.png")
    json.dump(report, open(out / "parts.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
