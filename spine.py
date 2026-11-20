import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure


def load(path, label):
    img = nib.as_closest_canonical(nib.load(path))
    m = np.asarray(img.dataobj) == label
    if not m.any():
        raise SystemExit(f"label {label} not found in {path}")
    m = np.pad(m, 3)
    idx = np.argwhere(m)
    lo, hi = idx.min(0) - 3, idx.max(0) + 4
    return m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.array(img.header.get_zooms()[:3], float)


def mesh(m, sp):
    smooth = ndimage.gaussian_filter(m.astype(float), 1)
    verts, faces, _, _ = measure.marching_cubes(smooth, 0.5, spacing=tuple(sp))
    return verts, faces


def cylinder(a, b, r, k=24):
    d = (b - a) / np.linalg.norm(b - a)
    u = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    ring = r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)
    verts = np.vstack([a + ring, b + ring])
    i = np.arange(k)
    j = (i + 1) % k
    faces = np.vstack([np.c_[i, j, k + i], np.c_[j, k + j, k + i]])
    return verts, faces


def write_stl(path, parts):
    tris = np.vstack([v[f] for v, f in parts])
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    data = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    data["n"] = n
    data["v"] = tris
    with open(path, "wb") as f:
        f.write(bytes(80))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(data.tobytes())


def inside(m, sp, q):
    i = np.round(q / sp).astype(int)
    return bool((i >= 0).all() and (i < m.shape).all() and m[tuple(i)])


def walk(m, sp, p, d):
    q = p.copy()
    while inside(m, sp, q + 0.5 * d):
        q = q + 0.5 * d
    return q


def line(m, sp, dist, p, d, r):
    entry = walk(m, sp, p, -d)
    tip = walk(m, sp, p, d) - 5 * d
    length = np.linalg.norm(tip - entry)
    path = [entry + s * d for s in np.arange(5, length, 0.5)]
    clear = min(dist[tuple(np.round(q / sp).astype(int))] for q in path) - r if path else -r
    return entry, tip, length, clear


def plan(m, sp, r=3.25):
    pts = np.argwhere(m) * sp
    cx = pts[:, 0].mean()
    xs = np.arange(m.shape[0]) * sp[0]
    midline = m[np.abs(xs - cx) < 5].any(axis=(0, 2))
    yb = np.flatnonzero(m.any(axis=(0, 2))).max()
    while yb > 0 and midline[yb - 1]:
        yb -= 1
    yl = yb - 1
    while yl > 0 and not midline[yl]:
        yl -= 1
    canal = range(yl + 1, yb)
    if len(canal) == 0:
        raise SystemExit("spinal canal not found")
    body = pts[pts[:, 1] >= yb * sp[1]]
    bc = body.mean(0)
    dist = ndimage.distance_transform_edt(m, sampling=sp)
    screws = []
    for name, side, sign in (("left", xs < cx, 1), ("right", xs >= cx, -1)):
        sub = m & side[:, None, None]
        area = sub.sum(axis=(0, 2))
        k = min((k for k in canal if area[k] > 0), key=lambda k: area[k])
        cut = np.argwhere(sub[:, k, :])
        c = cut.mean(0)
        p = np.array([c[0] * sp[0], k * sp[1], c[1] * sp[2]])
        best = None
        for a in range(-10, 46, 2):
            for b in range(-14, 15, 2):
                ra, rb = np.radians(a), np.radians(b)
                d = np.array([sign * np.sin(ra) * np.cos(rb), np.cos(ra) * np.cos(rb), np.sin(rb)])
                entry, tip, length, clear = line(m, sp, dist, p, d, r)
                if tip[1] >= bc[1] and (best is None or clear > best[4]):
                    best = (a, b, entry, tip, clear, length)
        if best is None:
            raise SystemExit(f"no trajectory found on the {name} side")
        a, b, entry, tip, clear, length = best
        screws.append({
            "side": name,
            "isthmus": p.round(2).tolist(),
            "entry": entry.round(2).tolist(),
            "tip": tip.round(2).tolist(),
            "length_mm": round(float(length), 1),
            "medial_angle_deg": a,
            "cranial_angle_deg": b,
            "pedicle_width_mm": round(float(np.ptp(cut[:, 0]) * sp[0] + sp[0]), 1),
            "pedicle_height_mm": round(float(np.ptp(cut[:, 1]) * sp[2] + sp[2]), 1),
            "clearance_mm": round(float(clear), 2),
            "breach": bool(clear < 0),
        })
    return {
        "voxel_mm": sp.round(3).tolist(),
        "vertebra_size_mm": (np.ptp(pts, 0) + sp).round(1).tolist(),
        "body_size_mm": (np.ptp(body, 0) + sp).round(1).tolist(),
        "screw_diameter_mm": 2 * r,
        "screws": screws,
    }


if __name__ == "__main__":
    mask_path, out = sys.argv[1], Path(sys.argv[2])
    label = int(sys.argv[3]) if len(sys.argv) > 3 else 22
    out.mkdir(parents=True, exist_ok=True)
    m, sp = load(mask_path, label)
    v, f = mesh(m, sp)
    res = plan(m, sp)
    write_stl(out / "vertebra.stl", [(v, f)])
    screws = [cylinder(np.array(s["entry"]), np.array(s["tip"]), res["screw_diameter_mm"] / 2) for s in res["screws"]]
    write_stl(out / "vertebra_with_screws.stl", [(v, f)] + screws)
    (out / "plan.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
