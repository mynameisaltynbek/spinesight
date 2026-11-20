import json
import sys

import cv2
import numpy as np

from common import FONT, LANDMARKS, board_pose, kabsch, load_camera, load_stl, open_camera, ray, read, stl_landmarks, text, triangulate


def landmark_sheet(tris, pts):
    v = tris.reshape(-1, 3)
    lo = v.min(0)
    s = 560 / np.ptp(v, 0).max()
    img = np.full((600, 1200, 3), 255, np.uint8)
    for a, b, ox, title in ((0, 1, 0, "top view, front up"), (1, 2, 600, "side view, front right")):
        q = ((v[:, [a, b]] - lo[[a, b]]) * s).astype(int)
        img[590 - q[:, 1], ox + 10 + q[:, 0]] = 150
        cv2.putText(img, title, (ox + 10, 25), FONT, 0.7, (0, 0, 0), 2)
        for i, p in enumerate(pts):
            c = ((p[[a, b]] - lo[[a, b]]) * s).astype(int)
            cv2.circle(img, (ox + 10 + c[0], 590 - c[1]), 9, (0, 0, 255), 2)
            cv2.putText(img, str(i + 1), (ox + 22 + c[0], 585 - c[1]), FONT, 0.8, (0, 0, 255), 2)
    cv2.imwrite("landmarks.png", img)


def click(frame, name):
    pt = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt[:] = [(x, y)]

    cv2.setMouseCallback("register", on_mouse)
    while True:
        view = frame.copy()
        text(view, f"click: {name}   enter: ok   s: not visible")
        if pt:
            cv2.circle(view, pt[0], 5, (0, 0, 255), -1)
        cv2.imshow("register", view)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and pt:
            return pt[0]
        if key == ord("s"):
            return None


cam = load_camera()
tris = load_stl(sys.argv[1] if len(sys.argv) > 1 else "../outputs/vertebra.stl")
model = stl_landmarks(tris)
landmark_sheet(tris, model)
print("landmarks.png shows the points to click:")
for i, name in enumerate(LANDMARKS):
    print(f"  {i + 1}. {name}")

cap = open_camera(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
views = []
while True:
    frame = read(cap)
    pose = board_pose(frame, cam)
    view = frame.copy()
    text(view, f"views {len(views)}   board {'ok' if pose else 'not found'}   space: take view   q: done")
    cv2.imshow("register", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and pose:
        views.append((frame, pose))
    if key == ord("q"):
        break
cap.release()

rays = {i: [] for i in range(len(LANDMARKS))}
for frame, (R, t) in views:
    for i, name in enumerate(LANDMARKS):
        uv = click(frame, f"{i + 1}. {name}")
        if uv:
            rays[i].append(ray(R, t, cam, uv))
cv2.destroyAllWindows()

idx = [i for i in rays if len(rays[i]) >= 2]
if len(idx) < 3:
    raise SystemExit("need at least 3 landmarks, each clicked in 2 or more views")
board = np.array([triangulate(*zip(*rays[i])) for i in idx])
R, t = kabsch(model[idx], board)
err = np.linalg.norm(model[idx] @ R.T + t - board, axis=1)
res = {
    "R": R.tolist(),
    "t": t.tolist(),
    "landmarks": [LANDMARKS[i] for i in idx],
    "error_mm": err.round(2).tolist(),
    "rms_mm": round(float(np.sqrt((err ** 2).mean())), 2),
}
json.dump(res, open("vertebra_pose.json", "w"), indent=2)
print(json.dumps(res, indent=2))
