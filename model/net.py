import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

IN_W, IN_H, STRIDE = 640, 352, 4
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class Net(nn.Module):
    def __init__(self, k, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(*list(r.children())[:-2])
        layers, c = [], 512
        for _ in range(3):
            layers += [nn.ConvTranspose2d(c, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
            c = 256
        self.head = nn.Sequential(*layers, nn.Conv2d(256, k, 1))

    def forward(self, x):
        return self.head(self.backbone(x))


def image_path(label_path):
    jpg = label_path.with_suffix(".jpg")
    return jpg if jpg.exists() else label_path.with_suffix(".png")


def prep(img):
    h, w = img.shape[:2]
    x = (cv2.resize(img, (IN_W, IN_H))[:, :, ::-1] / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)), np.array([IN_W / w, IN_H / h])


def keypoints_2d(label, kps):
    if "keypoints_2d" in label:
        return np.array(label["keypoints_2d"])
    uv, _ = cv2.projectPoints(kps, cv2.Rodrigues(np.array(label["R"]))[0], np.array(label["t"]), np.array(label["K"]), np.array(label["dist"]))
    return uv.reshape(-1, 2)


def target(uv, k, sigma=2.0):
    ys, xs = np.mgrid[:IN_H // STRIDE, :IN_W // STRIDE]
    t = np.zeros((k, IN_H // STRIDE, IN_W // STRIDE), np.float32)
    for j, (x, y) in enumerate(uv / STRIDE):
        t[j] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return t


def peak_offset(a):
    l, c, r = np.log(np.clip(a, 1e-6, None))
    den = l - 2 * c + r
    return float(np.clip(0.5 * (l - r) / den, -0.5, 0.5)) if den < 0 else 0.0


def decode(hm):
    k, h, w = hm.shape
    uv, conf = np.zeros((k, 2)), np.zeros(k)
    for j in range(k):
        y, x = np.unravel_index(hm[j].argmax(), (h, w))
        dx = peak_offset(hm[j, y, x - 1:x + 2]) if 0 < x < w - 1 else 0.0
        dy = peak_offset(hm[j, y - 1:y + 2, x]) if 0 < y < h - 1 else 0.0
        uv[j] = [x + dx, y + dy]
        conf[j] = hm[j, y, x]
    return uv * STRIDE, conf


def solve_pose(uv, conf, kps, K, dist, thr=0.1):
    ok = conf > thr
    if ok.sum() < 4:
        return None
    found, rvec, tvec, inl = cv2.solvePnPRansac(kps[ok], uv[ok], K, dist, reprojectionError=8, flags=cv2.SOLVEPNP_EPNP)
    if not found or inl is None or len(inl) < 4:
        return None
    i = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(kps[ok][i], uv[ok][i], K, dist, rvec, tvec)
    return cv2.Rodrigues(rvec)[0], tvec.ravel()


def errors(R, t, Rg, tg, pts, screws):
    out = {
        "rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(R.T @ Rg) - 1) / 2, -1, 1)))),
        "trans_mm": float(np.linalg.norm(t - tg)),
        "add_mm": float(np.linalg.norm(pts @ (R - Rg).T + t - tg, axis=1).mean()),
    }
    for s in screws:
        e, tip = np.array(s["entry"]), np.array(s["tip"])
        d, dg = (tip - e) @ R.T, (tip - e) @ Rg.T
        cos = d @ dg / np.linalg.norm(d) / np.linalg.norm(dg)
        out[f"{s['side']}_entry_mm"] = float(np.linalg.norm(e @ (R - Rg).T + t - tg))
        out[f"{s['side']}_angle_deg"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return out
