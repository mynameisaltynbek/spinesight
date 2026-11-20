import json

import cv2
import numpy as np

SQUARE_MM = 30.0
MARKER_MM = 22.0
BOARD = cv2.aruco.CharucoBoard((7, 5), SQUARE_MM, MARKER_MM, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
DETECTOR = cv2.aruco.CharucoDetector(BOARD)
LANDMARKS = ["spinous process tip", "left transverse process tip", "right transverse process tip", "front of the body"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} not found")
    return cap


def read(cap):
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera stopped")
    return frame


def text(img, s):
    cv2.putText(img, s, (10, 30), FONT, 0.8, (0, 0, 255), 2)


def board_points(frame):
    corners, ids, _, _ = DETECTOR.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < 6:
        return None, None
    return BOARD.matchImagePoints(corners, ids)


def board_pose(frame, cam):
    obj, img = board_points(frame)
    if obj is None:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam["K"], cam["dist"])
    return (cv2.Rodrigues(rvec)[0], tvec.ravel()) if ok else None


def load_camera(path="camera.json"):
    c = json.load(open(path))
    return {"K": np.array(c["K"]), "dist": np.array(c["dist"])}


def load_stl(path):
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    with open(path, "rb") as f:
        f.seek(80)
        n = int(np.frombuffer(f.read(4), "<u4")[0])
        return np.frombuffer(f.read(n * dt.itemsize), dt)["v"].astype(float)


def stl_landmarks(tris):
    v = tris.reshape(-1, 3)
    return np.array([v[v[:, 1].argmin()], v[v[:, 0].argmin()], v[v[:, 0].argmax()], v[v[:, 1].argmax()]])


def draw_outline(frame, tris, R, t, cam):
    p, _ = cv2.projectPoints(tris.reshape(-1, 3), cv2.Rodrigues(R)[0], t, cam["K"], cam["dist"])
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, list(np.round(p).astype(np.int32).reshape(-1, 3, 2)), 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)


def ray(R, t, cam, uv):
    n = cv2.undistortPoints(np.array([[uv]], float), cam["K"], cam["dist"]).ravel()
    return -R.T @ t, R.T @ np.array([n[0], n[1], 1.0])


def triangulate(centers, dirs):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centers, dirs):
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca
