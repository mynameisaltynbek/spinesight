import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "rig"))
from common import load_stl

SCALE = 0.5
NUM_KEYPOINTS = 10


def camera():
    path = ROOT / "rig" / "camera.json"
    if path.exists():
        c = json.load(open(path))
        K, size = np.array(c["K"]), c["size"]
    else:
        K, size = np.array([[1000.0, 0, 640], [0, 1000, 360], [0, 0, 1]]), [1280, 720]
    K = K.copy()
    K[:2] *= SCALE
    return K, (round(size[0] * SCALE), round(size[1] * SCALE))


def farthest_points(v, k):
    idx = [int(np.linalg.norm(v - v.mean(0), axis=1).argmax())]
    d = np.linalg.norm(v - v[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(v - v[idx[-1]], axis=1))
    return v[idx]


def render(img, tris, R, t, K, color, light, shine):
    cam = tris @ R.T + t
    n = np.cross(cam[:, 1] - cam[:, 0], cam[:, 2] - cam[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    center = cam.mean(1)
    n[(n * center).sum(1) > 0] *= -1
    view = -center / np.linalg.norm(center, axis=1, keepdims=True)
    half = (light + view) / np.linalg.norm(light + view, axis=1, keepdims=True)
    diffuse = np.clip(n @ light, 0, 1)
    spec = shine * np.clip((n * half).sum(1), 0, 1) ** 30
    shade = np.clip((0.3 + 0.7 * diffuse)[:, None] * color + 255 * spec[:, None], 0, 255)
    p = cam @ K.T
    pts = np.round(p[..., :2] / p[..., 2:] * 16).astype(np.int32)
    ids = np.zeros(img.shape[:2], np.int32)
    for i in np.argsort(-center[:, 2]):
        cv2.fillConvexPoly(img, pts[i], shade[i].tolist(), shift=4)
        cv2.fillConvexPoly(ids, pts[i], int(i) + 1, shift=4)
    return ids, center[:, 2]


def visible(kp_cam, uv, ids, depth):
    out = []
    for (x, y), z in zip(uv, kp_cam[:, 2]):
        i, j = int(round(y)), int(round(x))
        ok = 0 <= i < ids.shape[0] and 0 <= j < ids.shape[1] and ids[i, j] > 0
        out.append(bool(ok and abs(depth[ids[i, j] - 1] - z) < 4))
    return out


def background(rng, size, photos):
    w, h = size
    if photos and rng.random() < 0.8:
        img = cv2.imread(str(photos[rng.integers(len(photos))]))
        s = max(w / img.shape[1], h / img.shape[0]) * rng.uniform(1, 1.5)
        img = cv2.resize(img, (int(img.shape[1] * s) + 1, int(img.shape[0] * s) + 1))
        y, x = rng.integers(img.shape[0] - h + 1), rng.integers(img.shape[1] - w + 1)
        img = img[y:y + h, x:x + w]
        return np.ascontiguousarray(img[:, ::-1] if rng.random() < 0.5 else img)
    a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    g = np.linspace(0, 1, w)[None, :, None] if rng.random() < 0.5 else np.linspace(0, 1, h)[:, None, None]
    img = np.ascontiguousarray(np.broadcast_to(a + (b - a) * g, (h, w, 3)).astype(np.uint8))
    for _ in range(rng.integers(0, 15)):
        p = rng.integers(0, [w, h], (2, 2)).tolist()
        cv2.rectangle(img, p[0], p[1], rng.integers(0, 256, 3).tolist(), int(rng.choice([-1, 1, 3])))
    return img


def occlude(rng, img, ids, around):
    for _ in range(rng.integers(0, 3)):
        mask = np.zeros(ids.shape, np.uint8)
        c = (around + rng.normal(0, 60, 2)).astype(np.int32)
        if rng.random() < 0.5:
            color = (rng.uniform(60, 140), rng.uniform(100, 180), rng.uniform(150, 230))
            cv2.ellipse(mask, c.tolist(), rng.integers(20, 90, 2).tolist(), rng.uniform(0, 180), 0, 360, 255, -1)
        else:
            color = rng.integers(0, 256, 3)
            cv2.fillPoly(mask, [(c + rng.integers(-70, 70, (rng.integers(3, 7), 2))).astype(np.int32)], 255)
        img[mask > 0] = color
        ids[mask > 0] = 0


def degrade(rng, img):
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        k = np.zeros((9, 9))
        k[4] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((4, 4), rng.uniform(0, 180), 1), (9, 9))
        img = cv2.filter2D(img, -1, k / k.sum())
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30) + rng.normal(0, rng.uniform(0, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def sample(i, tris, kps, K, size, photos):
    rng = np.random.default_rng(i)
    R = Rotation.from_quat(rng.normal(size=4)).as_matrix()
    c = rng.uniform(250, 600) * np.linalg.inv(K) @ [rng.uniform(0.2, 0.8) * size[0], rng.uniform(0.2, 0.8) * size[1], 1]
    t = c - R @ tris.reshape(-1, 3).mean(0)
    img = background(rng, size, photos)
    light = rng.normal(size=3)
    light[2] = -abs(light[2])
    color = rng.uniform(150, 255) * rng.uniform(0.8, 1, 3)
    ids, depth = render(img, tris, R, t, K, color, light / np.linalg.norm(light), rng.uniform(0, 0.6))
    kp_cam = kps @ R.T + t
    uv = kp_cam @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    occlude(rng, img, ids, uv.mean(0))
    img = degrade(rng, img)
    label = {
        "R": R.tolist(),
        "t": t.tolist(),
        "K": K.tolist(),
        "dist": [0.0] * 5,
        "keypoints_2d": uv.round(2).tolist(),
        "visible": visible(kp_cam, uv, ids, depth),
    }
    return img, label, int(rng.integers(50, 96))


def sheet(paths):
    tiles = []
    for p in paths[:16]:
        img = cv2.imread(str(p))
        s = 320 / img.shape[1]
        img = cv2.resize(img, (320, round(img.shape[0] * s)))[:180]
        img = np.vstack([img, np.zeros((180 - img.shape[0], 320, 3), np.uint8)])
        lab = json.load(open(p.with_suffix(".json"))) if p.with_suffix(".json").exists() else {}
        for (x, y), v in zip(lab.get("keypoints_2d", []), lab.get("visible", [])):
            cv2.circle(img, (int(x * s), int(y * s)), 3, (0, 255, 0) if v else (0, 0, 255), -1)
        tiles.append(img)
    tiles += [np.zeros((180, 320, 3), np.uint8)] * (16 - len(tiles))
    return np.vstack([np.hstack(tiles[r * 4:r * 4 + 4]) for r in range(4)])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "synth_out"
    out.mkdir(exist_ok=True)
    tris = load_stl(ROOT / "outputs" / "vertebra.stl")
    kps = farthest_points(tris.reshape(-1, 3), NUM_KEYPOINTS)
    K, size = camera()
    photos = sorted(p for p in (ROOT / "synth" / "backgrounds").glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    json.dump({"keypoints_3d": kps.round(3).tolist(), "image_size": size}, open(out / "keypoints.json", "w"), indent=2)
    for i in range(n):
        img, label, quality = sample(i, tris, kps, K, size, photos)
        cv2.imwrite(str(out / f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        json.dump(label, open(out / f"{i:06d}.json", "w"))
        if i % 100 == 0:
            print(f"{i}/{n}")
    cv2.imwrite(str(out / "preview.jpg"), sheet(sorted(out.glob("[0-9]*.jpg"))))
    real = sorted((ROOT / "test_real").glob("*.png"))
    if real:
        cv2.imwrite(str(out / "preview_real.jpg"), sheet(real))
    print(f"done: {n} images in {out}, {len(photos)} background photos used")
